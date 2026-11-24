"""Loads the real functions of app.py (analyze_signal, simple_backtest, ...) without starting Streamlit or touching the network.
Only top-level imports / defs / simple assignments are executed; UI statements are skipped."""
import ast, os, sys, tempfile
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "app.py")) else os.path.dirname(HERE)   # works in tests/ or next to app.py
_NS = {}


def load(app_file="app.py"):
    key = (ROOT, app_file)
    if key in _NS:
        return _NS[key]
    os.environ.setdefault("ELITE_DATA_DIR", tempfile.mkdtemp(prefix="elite_reg_"))
    sys.path.insert(0, ROOT)
    for extra in os.environ.get("EXTRA_TEST_PATH", "").split(os.pathsep):
        if extra:
            sys.path.append(extra)
    for m in ("yfinance", "plotly", "plotly.graph_objects", "plotly.express", "streamlit"):
        try:
            __import__(m)
        except Exception:
            sys.modules[m] = MagicMock()
    sys.modules.setdefault("streamlit", MagicMock())
    src = open(os.path.join(ROOT, app_file), encoding="utf-8").read()
    ns = {"__name__": "appns"}
    for node in ast.parse(src).body:
        is_export = isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "_ec_export"
        if is_export or isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Try)):
            try:
                exec(compile(ast.Module([node], []), app_file, "exec"), ns)
            except BaseException:
                pass
    st = MagicMock()
    st.session_state = {}
    ns["st"] = st
    _NS[key] = ns
    return ns


def sample_frame(n=400, seed=3, start="2026-11-16 09:15", freq="5min", drift=0.0):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    c = 25000 + np.cumsum(rng.normal(drift, 8, n))
    return pd.DataFrame({"Open": c, "High": c + 6, "Low": c - 6, "Close": c, "Volume": rng.integers(1000, 5000, n)}, index=idx)
