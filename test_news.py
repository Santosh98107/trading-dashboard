"""News filter whole-word matching. Run: python -m pytest tests/test_news.py -q"""
import os, sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "app.py")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import _appns

NS = _appns.load()
T = NS["NEWS_HIGH_IMPACT_TERMS"]
find = NS["_news_terms_in"]


def test_original_substring_behaviour_is_kept_when_off():
    assert "war" in find("new software launched", T, False)
    assert "war" in find("profit warning issued", T, False)


def test_whole_word_mode_stops_false_hits_but_keeps_real_ones():
    for fake in ("new software launched", "moving toward growth", "he is a stalwart", "strong governance guidance-free"):
        assert "war" not in find(fake, T, True), fake
    assert "war" in find("war escalates in region", T, True)
    assert "war" in find("fears of war, oil jumps", T, True)
    assert "rate cut" in find("rbi announces rate cuts today", T, True)            # plural still counts
    assert "fraud" in find("bank fraud probe", T, True) and "sebi" in find("SEBI order".lower(), T, True)
    assert "reserve bank" in find("reserve bank of india says", T, True)
    assert "earnings" in find("q2 earnings beat", T, True)
    assert find("south indian bank stake approved", T, True) == []


def _plain_fn():
    """get_news_events without its st.cache_data decorator (streamlit is mocked in tests)."""
    import ast
    if "_plain_news" not in NS:
        src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "get_news_events")
        node.decorator_list = []
        ns2 = dict(NS)
        exec(compile(ast.Module([node], []), "app.py", "exec"), ns2)
        NS["_plain_news"] = ns2["get_news_events"]
    return NS["_plain_news"]


def _payload(title, desc):
    pub = (datetime.now(timezone.utc) - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    return {"data": [{"title": title, "description": desc, "published_at": pub, "source": "x.com", "url": "http://x", "entities": []}]}


def _run(word_match, title, desc):
    fn = _plain_fn()
    resp = MagicMock(status_code=200); resp.json.return_value = _payload(title, desc)
    fake_requests = MagicMock(); fake_requests.get.return_value = resp
    fake_requests.Timeout = TimeoutError
    fn.__globals__["requests"] = fake_requests
    return fn("tok", "NIFTY", provider="Marketaux", lookback_hours=6, article_limit=10, word_match=word_match)


def test_end_to_end_false_block_goes_away_and_real_block_stays():
    fake = ("Software firm wins order", "A new software contract was signed.")
    real = ("War fears hit markets", "Oil jumps as war escalates.")
    assert _run(False, *fake)["allow_trade"] is False          # old behaviour: 'war' inside 'software' blocks
    assert _run(True, *fake)["allow_trade"] is True
    assert _run(True, *real)["allow_trade"] is False
    assert _run(False, *real)["allow_trade"] is False
