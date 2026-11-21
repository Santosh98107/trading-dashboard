"""Pre-trade validation + dry-run broker. NO live broker adapter on purpose:
real-money order code cannot be tested here and must be reviewed with your broker/SEBI rules."""
import time

_LAST = {}
MAX_QTY_DEFAULT = 10000


def validate_order(symbol, side, qty, price, last_price=None, band_pct=2.0, max_qty=MAX_QTY_DEFAULT,
                   dup_window_sec=5, trading_enabled=True, now=None):
    now = time.time() if now is None else now
    if not trading_enabled:
        return False, "Trading halted (kill switch)"
    try:
        qty = int(qty); price = float(price)
    except Exception:
        return False, "Invalid qty/price"
    if side not in ("BUY", "SELL") or qty <= 0 or price <= 0:
        return False, "Invalid order"
    if qty > max_qty:
        return False, f"Quantity {qty} above cap {max_qty}"
    if last_price and abs(price - last_price) / last_price * 100 > band_pct:
        return False, f"Price {price} is more than {band_pct}% away from last {last_price}"
    key = (symbol, side, qty)
    if now - _LAST.get(key, 0) < dup_window_sec:
        return False, "Duplicate order within a few seconds blocked"
    _LAST[key] = now
    return True, "OK"


class DryRunBroker:
    """Records what WOULD be sent. Places nothing."""
    def __init__(self):
        self.log = []

    def place(self, symbol, side, qty, price):
        self.log.append({"ts": time.time(), "symbol": symbol, "side": side, "qty": qty, "price": price, "sent": False})
        return {"status": "DRY_RUN", "sent": False}
