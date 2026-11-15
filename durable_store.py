"""durable_store.py - keeps a copy of elite.db outside Streamlit Cloud's disposable disk.

How it works (the app's own SQLite code is not touched):
  * At startup, BEFORE the database is opened, restore_if_needed() copies the saved snapshot back when the local
    elite.db is missing (a redeploy / reboot wiped the disk).
  * At the end of every run, maybe_push() uploads a consistent snapshot in a background thread (at most once per
    PUSH_INTERVAL_SEC, and only when something changed).
Backends (chosen from secrets / environment variables):
  * Supabase Storage (HTTPS only, needs the `requests` package Streamlit already ships):
        SUPABASE_URL, SUPABASE_SERVICE_KEY, optional SUPABASE_BUCKET (default "elite-backups")
  * A folder (a persistent volume, or for tests):  DURABLE_DIR
Nothing is configured -> every function is a harmless no-op and the app behaves exactly as before.
"""
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone

PUSH_INTERVAL_SEC = 30
MAX_BYTES = 45 * 1024 * 1024          # Supabase free plan allows 50 MB per object
OBJ_CURRENT = "elite_current.db"
OBJ_META = "elite_current.json"
HTTP_TIMEOUT = 20

_LOCK = threading.RLock()
_STATE = {
    "checked": False, "restore_blocked": False, "last_push_try": 0.0, "last_sha": None, "last_daily": "",
    "pushing": False, "last_ok": "", "last_error": "", "restored": "", "remote_info": "", "pushes": 0,
}


# ---------------------------------------------------------------- configuration
def normalize_supabase_url(url):
    """Accept whatever was pasted (…supabase.co, …supabase.co/, …supabase.co/rest/v1/, no https://) and keep only
    scheme://host. The Storage API lives under /storage/v1 of that host; an extra /rest/v1 path makes every call 404."""
    from urllib.parse import urlparse
    u = str(url or "").strip().strip('"').strip("'")
    if not u:
        return ""
    if "://" not in u:
        u = "https://" + u
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}" if p.netloc else u


def _config_path():
    return os.path.join(os.environ.get("ELITE_DATA_DIR", "elite_data"), "durable_config.json")


def load_local_config():
    """Settings typed into the app (Settings -> Permanent storage). Lives on the local disk only."""
    try:
        with open(_config_path(), "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def save_local_config(url=None, key=None, bucket=None, enabled=None):
    cfg = load_local_config()
    if url is not None:
        cfg["SUPABASE_URL"] = normalize_supabase_url(url)
    if key is not None and str(key).strip():
        cfg["SUPABASE_SERVICE_KEY"] = str(key).strip()      # an empty box keeps the saved key
    if bucket is not None:
        cfg["SUPABASE_BUCKET"] = str(bucket).strip()
    if enabled is not None:
        cfg["enabled"] = bool(enabled)
    os.makedirs(os.path.dirname(_config_path()), exist_ok=True)
    tmp = _config_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    os.replace(tmp, _config_path())


def is_enabled():
    return bool(load_local_config().get("enabled", True))


def _secret(name):
    val = os.environ.get(name)
    if val:
        return str(val).strip()
    try:
        import streamlit as st
        v = st.secrets.get(name)          # raises when no secrets file exists
        if v:
            return str(v).strip()
    except Exception:
        pass
    return str(load_local_config().get(name, "") or "").strip()      # typed into the Settings page


def default_db_file():
    return os.path.join(os.environ.get("ELITE_DATA_DIR", "elite_data"), "elite.db")


def _backend(ignore_enabled=False):
    if not ignore_enabled and not is_enabled():
        return None                       # switched OFF in Settings
    folder = _secret("DURABLE_DIR")
    if folder:
        return FolderBackend(folder)
    url, key = _secret("SUPABASE_URL"), _secret("SUPABASE_SERVICE_KEY")
    if url and key:
        return SupabaseBackend(normalize_supabase_url(url), key, _secret("SUPABASE_BUCKET") or "elite-backups")
    return None


class FolderBackend:
    name = "folder"

    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def put(self, obj, data):
        tmp = os.path.join(self.folder, obj + ".part")
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, os.path.join(self.folder, obj))

    def get(self, obj):
        path = os.path.join(self.folder, obj)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()


class SupabaseBackend:
    name = "supabase"

    def __init__(self, url, key, bucket):
        self.base = url.rstrip("/") + "/storage/v1"
        self.bucket = bucket
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        self._bucket_ok = False

    def _ensure_bucket(self):
        if self._bucket_ok:
            return
        import requests
        r = requests.post(self.base + "/bucket", headers={**self.headers, "Content-Type": "application/json"},
                          data=json.dumps({"id": self.bucket, "name": self.bucket, "public": False}), timeout=HTTP_TIMEOUT)
        # 200 = created; 400/409 = already exists. Anything else is surfaced by the next call.
        self._bucket_ok = True

    def put(self, obj, data):
        import requests
        url = f"{self.base}/object/{self.bucket}/{obj}"
        hdr = {**self.headers, "x-upsert": "true", "Content-Type": "application/octet-stream"}
        r = requests.post(url, headers=hdr, data=data, timeout=HTTP_TIMEOUT)
        if r.status_code in (400, 404) and "not found" in r.text.lower():
            self._ensure_bucket()
            r = requests.post(url, headers=hdr, data=data, timeout=HTTP_TIMEOUT)
        if r.status_code >= 300:
            raise RuntimeError(f"Supabase upload failed ({r.status_code}): {r.text[:200]}")

    def get(self, obj):
        import requests
        r = requests.get(f"{self.base}/object/authenticated/{self.bucket}/{obj}", headers=self.headers, timeout=HTTP_TIMEOUT)
        if r.status_code == 404 or (r.status_code == 400 and "not found" in r.text.lower()):
            return None
        if r.status_code >= 300:
            raise RuntimeError(f"Supabase download failed ({r.status_code}): {r.text[:200]}")
        return r.content


# ---------------------------------------------------------------- helpers
def _now_text():
    return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%d-%m-%Y %H:%M:%S IST")


def _snapshot_bytes(db_file):
    """Consistent copy of the live SQLite database (SQLite online-backup API, safe while the app writes)."""
    src = sqlite3.connect(db_file, timeout=20)
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        dst = sqlite3.connect(tmp.name)
        with dst:
            src.backup(dst)
        dst.close()
        with open(tmp.name, "rb") as fh:
            return fh.read()
    finally:
        src.close()
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def _is_healthy_db(data):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        con = sqlite3.connect(tmp.name)
        ok = con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        tables = con.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        con.close()
        return bool(ok and tables > 0)
    except Exception:
        return False
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def _write_db_file(db_file, data):
    os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)
    for suffix in ("-wal", "-shm"):
        try:
            os.remove(db_file + suffix)
        except Exception:
            pass
    tmp = db_file + ".restoring"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, db_file)


# ---------------------------------------------------------------- restore
def restore_if_needed(db_file=None):
    """Call once at startup, before anything opens the database. Cheap after the first call in a process."""
    with _LOCK:
        if _STATE["checked"]:
            return
        _STATE["checked"] = True
        db_file = db_file or default_db_file()
        backend = _backend()
        if backend is None:
            return
        if os.path.exists(db_file) and os.path.getsize(db_file) > 0:
            return                      # disk survived: nothing to restore
        try:
            data = backend.get(OBJ_CURRENT)
            if data is None:
                return                  # first ever run: nothing saved yet, the app starts fresh and will push
            if not _is_healthy_db(data):
                raise RuntimeError("the saved snapshot failed the integrity check")
            _write_db_file(db_file, data)
            _STATE["restored"] = f"Restored {len(data) // 1024} KB from {backend.name} at {_now_text()}"
        except Exception as exc:
            # Do NOT let a fresh empty database overwrite a good remote copy later.
            _STATE["restore_blocked"] = True
            _STATE["last_error"] = f"Restore failed: {exc}"


def force_restore(db_file=None):
    """Replace the live database with the saved snapshot (Settings button, or retry after a failed startup restore)."""
    db_file = db_file or default_db_file()
    backend = _backend()
    if backend is None:
        return False, "Permanent storage is not configured."
    try:
        data = backend.get(OBJ_CURRENT)
        if data is None:
            return False, "Nothing has been saved to permanent storage yet."
        if not _is_healthy_db(data):
            return False, "The saved snapshot failed the integrity check; it was not applied."
        if os.path.exists(db_file):
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            tmp.write(data)
            tmp.close()
            try:
                up = sqlite3.connect(tmp.name)
                live = sqlite3.connect(db_file, timeout=20)
                up.backup(live)
                live.close()
                up.close()
            finally:
                os.remove(tmp.name)
        else:
            _write_db_file(db_file, data)
        with _LOCK:
            _STATE["restore_blocked"] = False
            _STATE["last_sha"] = hashlib.sha256(data).hexdigest()
            _STATE["restored"] = f"Restored {len(data) // 1024} KB from {backend.name} at {_now_text()}"
            _STATE["last_error"] = ""
        return True, _STATE["restored"]
    except Exception as exc:
        return False, f"Restore failed: {exc}"


# ---------------------------------------------------------------- push
def push_now(db_file=None, force=False):
    """Upload a snapshot now. Returns (ok, message)."""
    db_file = db_file or default_db_file()
    backend = _backend()
    if backend is None:
        return False, "Permanent storage is not configured."
    if _STATE["restore_blocked"]:
        return False, "Not saving: the startup restore failed, and saving now could overwrite your good copy. Press 'Restore from cloud' first."
    if not os.path.exists(db_file):
        return False, "No local database yet."
    try:
        data = _snapshot_bytes(db_file)
        if len(data) > MAX_BYTES:
            raise RuntimeError(f"database is {len(data) // 1024 // 1024} MB, over the {MAX_BYTES // 1024 // 1024} MB limit")
        sha = hashlib.sha256(data).hexdigest()
        if not force and sha == _STATE["last_sha"]:
            return True, "No change since the last save."
        backend.put(OBJ_CURRENT, data)
        meta = {"sha": sha, "bytes": len(data), "saved_at": _now_text()}
        backend.put(OBJ_META, json.dumps(meta).encode())
        today = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y%m%d")
        if _STATE["last_daily"] != today:
            backend.put(f"elite_daily_{today}.db", data)      # one rollback point per day
            _STATE["last_daily"] = today
        with _LOCK:
            _STATE["last_sha"] = sha
            _STATE["last_ok"] = f"{_now_text()} ({len(data) // 1024} KB)"
            _STATE["last_error"] = ""
            _STATE["pushes"] += 1
        return True, "Saved to permanent storage at " + _STATE["last_ok"]
    except Exception as exc:
        with _LOCK:
            _STATE["last_error"] = f"Save failed: {exc}"
        return False, _STATE["last_error"]


def _push_thread(db_file):
    try:
        if _STATE["restore_blocked"]:
            ok, _ = force_restore(db_file)          # keep retrying the restore; never push until it works
        else:
            push_now(db_file)
    finally:
        _STATE["pushing"] = False


def maybe_push(db_file=None):
    """Call at the end of every run. Starts at most one background upload per PUSH_INTERVAL_SEC."""
    try:
        if _backend() is None:
            return
        now = time.time()
        with _LOCK:
            if _STATE["pushing"] or now - _STATE["last_push_try"] < PUSH_INTERVAL_SEC:
                return
            _STATE["pushing"] = True
            _STATE["last_push_try"] = now
        threading.Thread(target=_push_thread, args=(db_file or default_db_file(),), daemon=True).start()
    except Exception as exc:
        _STATE["pushing"] = False
        _STATE["last_error"] = f"Scheduler error: {exc}"


def test_connection():
    """Write a tiny object and read it back, so a wrong URL / key / bucket shows a clear message."""
    backend = _backend(ignore_enabled=True)
    if backend is None:
        return False, "Enter the Supabase URL and the service_role key first (or add them to Secrets)."
    try:
        stamp = _now_text().encode()
        backend.put("elite_ping.txt", stamp)
        back = backend.get("elite_ping.txt")
        if back != stamp:
            return False, "Connected, but the test file did not read back correctly."
        return True, "Connection works: a test file was written to and read from the bucket."
    except Exception as exc:
        msg = str(exc)
        if "(401)" in msg or "(403)" in msg or "Invalid API key" in msg or "Invalid JWT" in msg:
            hint = " (the key is wrong - use the service_role key, not anon)"
        elif "(404)" in msg and "Bucket not found" not in msg:
            hint = " (the URL is probably wrong - use only https://<project>.supabase.co, without /rest/v1)"
        elif "Name or service not known" in msg or "getaddrinfo" in msg or "Max retries" in msg or "Invalid URL" in msg:
            hint = " (the URL looks wrong - it should look like https://xxxx.supabase.co)"
        else:
            hint = ""
        return False, f"Connection failed: {msg[:200]}{hint}"


def status():
    backend = _backend()
    return {
        "enabled": is_enabled(),
        "has_credentials": _backend(ignore_enabled=True) is not None,
        "configured": backend is not None,
        "backend": backend.name if backend else "none",
        "last_ok": _STATE["last_ok"], "last_error": _STATE["last_error"], "restored": _STATE["restored"],
        "blocked": _STATE["restore_blocked"], "pushes": _STATE["pushes"], "pushing": _STATE["pushing"],
    }
