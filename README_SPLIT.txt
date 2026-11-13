FOLDER LAYOUT  (all files must sit in the SAME folder / GitHub repo root)
  app.py                    <- main app (replaces your old "app (2).py")
  database.py               <- NEW: files, users/password, backups, journal, paper & risk state
  risk_manager.py           <- NEW: risk locks, paper trading, stop-loss / targets / trailing
  ai_engine.py              <- NEW: whole AI Lab
  institutional_upgrades.py <- your existing file, unchanged (keep it in the same folder)
  requirements.txt          <- your existing file, unchanged

STEPS
 1. In your GitHub repo upload these 4 files: app.py, database.py, risk_manager.py, ai_engine.py
 2. Streamlit Cloud > App settings > Main file path = app.py  (if your old file name was different)
 3. Delete the old "app (2).py" only after the new app works.
 4. Reboot the app. Sidebar shows: App build: 2026-11-13-SPLIT-MODULES
