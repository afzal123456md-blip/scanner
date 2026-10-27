"""
CoinDCX futures scanner + Telegram companion for the 5-Pillar Quick-Limit Precision System v2.0.

Run:   python bot.py            (24/7 loop)
       python bot.py --check    (test CoinDCX API + Telegram, then exit)
       python bot.py --dry      (one scan, print alerts instead of sending)
       python bot.py --once     (one full pass: commands + scan + monitor, for GitHub Actions)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

import strategy as S

HERE = Path(__file__).parent
CANDLE_URL = "https://public.coindcx.com/market_data/candlesticks"
TICKER_URL = "https://api.coindcx.com/exchange/ticker"
RES_SEC = {"5": 300, "60": 3600}


# --------------------------------------------------------------------------
# env / config
# --------------------------------------------------------------------------
def load_env():
    p = HERE / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env_cfg() -> S.Config:
    cfg = S.Config()
    g = os.environ.get
    cfg.usdt_inr = float(g("USDT_INR", cfg.usdt_inr))
    cfg.leverage = int(g("LEVERAGE", cfg.leverage))
    cfg.fee_in = float(g("FEE_IN", cfg.fee_in))
    cfg.fee_out = float(g("FEE_OUT", cfg.fee_out))
    cfg.min_score = int(g("MIN_SCORE", cfg.min_score))
    cfg.tp2_pct = float(g("TP2_PCT", cfg.tp2_pct))
    if g("QTY_STEPS"):
        cfg.qty_step = {k: float(v) for k, v in (x.split(":") for x in g("QTY_STEPS").split(","))}
    return cfg


# --------------------------------------------------------------------------
# state (trade log + P&L)
# --------------------------------------------------------------------------
class State:
    def __init__(self, path: Path, initial: float, target: float, usdt_inr: float):
        self.path = path
        if path.exists():
            self.d = json.loads(path.read_text())
        else:
            self.d = dict(initial_inr=initial, current_inr=initial, peak_inr=initial, target_inr=target,
                          usdt_inr=usdt_inr, realized_pnl_inr=0.0, counter=0, trades={}, paused=False,
                          cooldown_until=0, last_error_ts=0)
            self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str))
        tmp.replace(self.path)

    def new_id(self) -> str:
        self.d["counter"] += 1
        return f"S{self.d['counter']}"

    def open_trade(self):
        for tid, t in self.d["trades"].items():
            if t["status"] in ("staged", "filled"):
                return tid, t
        return None, None

    def status_text(self) -> str:
        tid, t = self.open_trade()
        if not t:
            return "Flat"
        return f"{'In Position' if t['status'] == 'filled' else 'Active Limit Staged'} ({tid} {t['coin']} {t['side']})"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
http = requests.Session()


def fetch_candles(pair: str, res: str, bars: int) -> pd.DataFrame:
    to = int(time.time())
    frm = to - bars * RES_SEC[res]
    r = http.get(CANDLE_URL, params={"pair": pair, "from": frm, "to": to, "resolution": res, "pcode": "f"}, timeout=20)
    r.raise_for_status()
    j = r.json()
    rows = j["data"] if isinstance(j, dict) else j
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no candles for {pair} {res}")
    t = df["time"].astype("int64")
    unit = "ms" if t.iloc[0] > 1e11 else "s"
    df.index = pd.to_datetime(t, unit=unit, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[~df.index.duplicated()].sort_index()


def closed_only(df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(seconds=seconds) <= now]


def get_usdt_inr(default: float) -> float:
    try:
        for m in http.get(TICKER_URL, timeout=15).json():
            if m.get("market") == "USDTINR":
                return float(m["last_price"])
    except Exception:
        pass
    return default


def load_events() -> list[S.Event]:
    p = HERE / "events.json"
    out = []
    if p.exists():
        for e in json.loads(p.read_text()):
            out.append(S.Event(e["name"], datetime.fromisoformat(e["time"]).astimezone(S.IST), e.get("impact", "high")))
    return out


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------
class Telegram:
    def __init__(self, token: str, chat_id: str, dry: bool = False):
        self.base = f"https://api.telegram.org/bot{token}"
        self.chat_id = str(chat_id)
        self.dry = dry
        self.offset = 0

    def send(self, text: str):
        if self.dry:
            print("\n" + "=" * 60 + "\n" + text + "\n" + "=" * 60)
            return
        chunks, cur = [], ""
        for para in text.split("\n\n"):
            if len(cur) + len(para) + 2 > 3800:
                chunks.append(cur)
                cur = para
            else:
                cur = f"{cur}\n\n{para}" if cur else para
        chunks.append(cur)
        for c in chunks:
            r = http.post(f"{self.base}/sendMessage", json={"chat_id": self.chat_id, "text": c}, timeout=20)
            if not r.ok:
                print("telegram error", r.text)

    def updates(self, timeout: int = 20):
        if self.dry:
            time.sleep(1)
            return []
        r = http.get(f"{self.base}/getUpdates", params={"offset": self.offset, "timeout": timeout}, timeout=timeout + 10)
        out = []
        for u in r.json().get("result", []):
            self.offset = u["update_id"] + 1
            m = u.get("message") or {}
            if str(m.get("chat", {}).get("id")) == self.chat_id and m.get("text"):
                out.append(m["text"].strip())
        return out


# --------------------------------------------------------------------------
# bot
# --------------------------------------------------------------------------
class Bot:
    def __init__(self, dry: bool = False):
        load_env()
        self.cfg = env_cfg()
        self.dry = dry
        capital = float(os.environ.get("CAPITAL_INR", 34733))
        target = float(os.environ.get("TARGET_INR", 100000))
        self.symbols = [x.strip() for x in os.environ.get(
            "SYMBOLS", "B-BTC_USDT,B-ETH_USDT,B-SOL_USDT,B-XRP_USDT,B-DOGE_USDT,B-BNB_USDT,B-LINK_USDT,B-AVAX_USDT,B-ADA_USDT").split(",")]
        self.scan_s = int(os.environ.get("SCAN_SECONDS", 300))
        self.expiry_h = float(os.environ.get("UNFILLED_EXPIRY_H", 6))
        self.alert_valid_min = 45
        self.symbol_cooldown_h = 6
        self.loss_cooldown_h = 2
        self.max_alerts = 2
        self.state = State(HERE / "state.json", capital, target, self.cfg.usdt_inr)
        self.tg = Telegram(os.environ.get("TELEGRAM_TOKEN", "dry"), os.environ.get("TELEGRAM_CHAT_ID", "0"), dry)
        self.now = lambda: datetime.now(S.IST)

    # ---- helpers
    def refresh_cfg(self):
        self.cfg.usdt_inr = get_usdt_inr(self.state.d.get("usdt_inr", self.cfg.usdt_inr))
        self.state.d["usdt_inr"] = self.cfg.usdt_inr
        self.cfg.capital_inr = self.state.d["current_inr"]

    def dash(self) -> str:
        return S.dashboard(self.state.d, self.state.status_text())

    # ---- scanning
    def scan(self):
        self.refresh_cfg()
        now = self.now()
        events = load_events()
        why = S.gate(now, self.cfg, events)
        if why:
            print(f"[{now:%H:%M}] gate: {why}")
            return
        d = self.state.d
        if d["paused"]:
            return
        if self.state.open_trade()[0]:
            print(f"[{now:%H:%M}] trade open - scanner idle (max 1 open trade)")
            return
        if time.time() < d.get("cooldown_until", 0):
            print(f"[{now:%H:%M}] post-loss cooldown active (no revenge trading)")
            return
        # expire stale alerts
        for tid, t in d["trades"].items():
            if t["status"] == "alerted" and time.time() - t["alert_ts"] > self.alert_valid_min * 60:
                t["status"] = "expired"
        refs = {}
        for key, pair in (("BTC", "B-BTC_USDT"), ("ETH", "B-ETH_USDT")):
            try:
                refs[key] = closed_only(fetch_candles(pair, "60", 400), 3600)
            except Exception as e:
                print("ref fetch failed", key, e)
        found = []
        self.last_scan_ok = 0
        for sym in self.symbols:
            try:
                d1h = refs.get(sym.split("-")[1].split("_")[0]) if sym in ("B-BTC_USDT", "B-ETH_USDT") else None
                if d1h is None:
                    d1h = closed_only(fetch_candles(sym, "60", 400), 3600)
                d5 = fetch_candles(sym, "5", 360)
                d15 = S.resample(d5, "15min")
                d4h = S.resample(d1h, "4h")
                d4h = d4h.iloc[:-1] if len(d4h) else d4h  # drop forming 4H bar
                coin = sym.split("-")[1].split("_")[0]
                s, reason = S.evaluate(sym, d15, d1h, d4h, refs, now, self.cfg, events)
                self.last_scan_ok += 1
                print(f"[{now:%H:%M}] {coin}: {reason}" + (f" score {s.score}" if s else ""))
                if s and not self.recently_alerted(s):
                    found.append(s)
            except Exception as e:
                print("scan error", sym, e)
            time.sleep(0.3)
        if not found:
            return
        found = S.rank_setups(found)[: self.max_alerts]
        for s in found:
            tid = self.state.new_id()
            d["trades"][tid] = dict(id=tid, status="alerted", alert_ts=time.time(), coin=s.coin, symbol=s.symbol,
                                    side=s.side, t1=s.t1, t2=s.t2, blended=s.blended, sl=s.sl, tp1=s.tp1, tp2=s.tp2,
                                    risk_inr=s.risk_inr, score=s.score, rank=s.rank, notified=[])
            self.state.save()
            self.tg.send(S.render_alert(s, tid, self.cfg, d, now, events))

    def recently_alerted(self, s: S.Setup) -> bool:
        for t in self.state.d["trades"].values():
            if t["symbol"] == s.symbol and t["side"] == s.side and time.time() - t["alert_ts"] < self.symbol_cooldown_h * 3600:
                return True
        return False

    # ---- monitoring of staged / filled trades
    def monitor(self):
        tid, t = self.state.open_trade()
        if not t:
            return
        now_ts = time.time()
        short = t["side"] == "short"
        try:
            c = fetch_candles(t["symbol"], "5", 300)
        except Exception as e:
            print("monitor fetch failed", e)
            return
        since = pd.Timestamp(t.get("filled_ts") or t["staged_ts"], unit="s", tz="UTC")
        c_all = c[c.index >= pd.Timestamp(t["staged_ts"], unit="s", tz="UTC")]
        c = c[c.index >= since]
        if c.empty:
            return
        hi, lo = c["high"].max(), c["low"].min()
        hi_all, lo_all = c_all["high"].max(), c_all["low"].min()
        touched = lambda lvl, h, l: (h >= lvl) if short else (l <= lvl)
        reached = lambda lvl, h, l: (l <= lvl) if short else (h >= lvl)

        def note(key, msg):
            if key not in t["notified"]:
                t["notified"].append(key)
                self.state.save()
                self.tg.send(f"[{tid}] {msg}\n\n{self.dash()}")

        if t["status"] == "staged":
            if touched(t["t1"], hi_all, lo_all):
                note("t1", f"T1 {S.fpx(t['t1'])} was touched on {t['coin']} - check CoinDCX for a fill, then reply /filled {tid}.")
            if touched(t["t2"], hi_all, lo_all):
                note("t2", f"T2 {S.fpx(t['t2'])} was touched - both tranches probably filled. Verify in CoinDCX.")
            if self.expiry_h > 0 and now_ts - t["staged_ts"] > self.expiry_h * 3600 and "t1" not in t["notified"]:
                note("expiry", f"Unfilled for {self.expiry_h:.0f}h - setup is stale. Cancel the resting orders, then /cancel {tid}.")
            if touched(t["sl"], hi_all, lo_all) and "t1" not in t["notified"]:
                note("sl_unfilled", f"Price hit the SL level before T1 filled - setup invalid. Cancel orders and /cancel {tid}.")
        else:
            t2_hit = touched(t["t2"], hi_all, lo_all)
            avg = t["blended"] if t2_hit else t["t1"]
            fee = self.cfg.fee_in + self.cfg.fee_out
            be_trig = avg * (1 - self.cfg.breakeven_pct / 100) if short else avg * (1 + self.cfg.breakeven_pct / 100)
            be_stop = avg * (1 - fee) if short else avg * (1 + fee)
            if reached(be_trig, hi, lo):
                note("be", f"+0.75% reached on {t['coin']}. MOVE SL TO BREAKEVEN + FEES now: ~{S.fpx(be_stop)}.")
            if reached(t["tp1"], hi, lo):
                note("tp1", f"TP1 {S.fpx(t['tp1'])} touched - confirm the 70% filled. Log result with /closed {tid} <pnl_inr> when flat.")
            if touched(t["sl"], hi, lo):
                note("sl", f"SL {S.fpx(t['sl'])} level touched - confirm stop executed, then /closed {tid} <pnl_inr>.")
            if now_ts - t["filled_ts"] > self.cfg.time_stop_h * 3600 and "tp1" not in t["notified"]:
                note("time", f"6-HOUR KILL SWITCH: no TP1 within 6h of fill. Close {t['coin']} at breakeven or minor profit, then /closed {tid} <pnl_inr>.")

    # ---- Telegram commands
    def handle(self, text: str):
        parts = text.split()
        cmd = parts[0].lower().lstrip("/").split("@")[0]
        d = self.state.d
        arg = parts[1].upper() if len(parts) > 1 else None
        tr = d["trades"].get(arg) if arg else None
        if cmd in ("start", "help"):
            self.tg.send("Commands:\n/status - dashboard\n/staged S1 - orders are live\n/filled S1 - T1 filled\n"
                         "/closed S1 -125.5 - trade finished, P&L in INR\n/cancel S1\n/balance 34733 - sync account value (INR)\n"
                         "/scan - scan now\n/pause  /resume\n/log - recent trades")
        elif cmd == "status":
            self.tg.send(f"Scanner: {'PAUSED' if d['paused'] else 'running'}\n\n{self.dash()}")
        elif cmd == "staged" and tr:
            for o in d["trades"].values():
                if o["status"] == "alerted" and o["id"] != tr["id"]:
                    o["status"] = "superseded"
            tr["status"], tr["staged_ts"] = "staged", time.time()
            self.state.save()
            self.tg.send(f"{tr['id']} logged as STAGED. I'm watching {tr['coin']} for fills, +0.75% breakeven, TP/SL and the 6h kill switch.\n\n{self.dash()}")
        elif cmd == "filled" and tr:
            tr["status"], tr["filled_ts"] = "filled", time.time()
            tr.setdefault("staged_ts", time.time())
            self.state.save()
            self.tg.send(f"{tr['id']} logged as IN POSITION. 6h clock started.\n\n{self.dash()}")
        elif cmd == "cancel" and tr:
            tr["status"] = "cancelled"
            self.state.save()
            self.tg.send(f"{tr['id']} cancelled. Scanner is live again.\n\n{self.dash()}")
        elif cmd == "closed" and tr and len(parts) > 2:
            pnl = float(parts[2].replace("₹", "").replace(",", ""))
            tr["status"], tr["pnl_inr"] = "closed", pnl
            d["current_inr"] += pnl
            d["realized_pnl_inr"] += pnl
            d["peak_inr"] = max(d["peak_inr"], d["current_inr"])
            if pnl < 0:
                d["cooldown_until"] = time.time() + self.loss_cooldown_h * 3600
            self.state.save()
            extra = f"\nLoss logged. Scanner pauses {self.loss_cooldown_h}h - no revenge trading, same size next time." if pnl < 0 else ""
            self.tg.send(f"{tr['id']} closed: {S.inr(pnl, True)}.{extra}\n\n{self.dash()}")
        elif cmd == "balance" and len(parts) > 1:
            d["current_inr"] = float(parts[1].replace(",", ""))
            d["peak_inr"] = max(d["peak_inr"], d["current_inr"])
            self.state.save()
            self.tg.send(f"Account value synced.\n\n{self.dash()}")
        elif cmd == "pause":
            d["paused"] = True
            self.state.save()
            self.tg.send("Scanner paused.")
        elif cmd == "resume":
            d["paused"] = False
            self.state.save()
            self.tg.send("Scanner resumed.")
        elif cmd == "scan":
            self.tg.send("Scanning now...")
            self.scan()
        elif cmd == "log":
            rows = list(d["trades"].values())[-8:]
            self.tg.send("\n".join(f"{t['id']} {t['coin']} {t['side']} {t['status']} score {t['score']}"
                                   + (f" pnl {t['pnl_inr']:+.0f}" if "pnl_inr" in t else "") for t in rows) or "No trades yet.")
        else:
            self.tg.send("Unknown command or trade id. /help")

    # ---- single pass for scheduled hosts (GitHub Actions)
    def once(self):
        d = self.state.d
        self.tg.offset = d.get("tg_offset", 0)
        msgs = self.tg.updates(timeout=0)
        d["tg_offset"] = self.tg.offset
        self.state.save()
        if not d.get("announced"):
            d["announced"] = True
            self.state.save()
            self.tg.send("Scanner online (scheduled mode: replies and alerts can lag a few minutes). "
                         "Send /help for commands.\n\n" + self.dash())
        for msg in msgs:
            try:
                self.handle(msg)
            except Exception as e:
                self.tg.send(f"Command error: {e}")
        self.last_scan_ok = None
        self.scan()
        if self.last_scan_ok == 0 and time.time() - d.get("last_error_ts", 0) > 3600:
            d["last_error_ts"] = time.time()
            self.state.save()
            self.tg.send("WARNING: could not load CoinDCX candles for any pair. The host may be blocked or CoinDCX is down. "
                         "Run the workflow in 'check' mode and read the log. No alerts can be produced until this is fixed.")
        self.monitor()

    # ---- main loop
    def run(self):
        self.tg.send("Scanner online. Strategy: 5-Pillar Quick-Limit v2.0. Send /help for commands.\n\n" + self.dash())
        next_scan = 0.0
        next_monitor = 0.0
        while True:
            try:
                for msg in self.tg.updates(timeout=20):
                    try:
                        self.handle(msg)
                    except Exception as e:
                        self.tg.send(f"Command error: {e}")
                if time.time() >= next_scan:
                    self.scan()
                    next_scan = time.time() + self.scan_s
                if time.time() >= next_monitor:
                    self.monitor()
                    next_monitor = time.time() + 60
            except Exception:
                traceback.print_exc()
                if time.time() - self.state.d.get("last_error_ts", 0) > 3600:
                    self.state.d["last_error_ts"] = time.time()
                    self.state.save()
                    try:
                        self.tg.send("Scanner hit an error (see server logs). It keeps running.")
                    except Exception:
                        pass
                time.sleep(15)


def check():
    load_env()
    print("CoinDCX candles:", end=" ")
    try:
        df = fetch_candles("B-BTC_USDT", "60", 50)
        print(f"OK ({len(df)} bars, last close {df['close'].iloc[-1]})")
    except Exception as e:
        print("FAILED", e)
    print("USDT/INR:", get_usdt_inr(0))
    tok, cid = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if tok and cid:
        Telegram(tok, cid).send("Connection test OK - your scanner can reach you here.")
        print("Telegram test message sent")
    else:
        print("Telegram: set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--once" in sys.argv:
        Bot().once()
    elif "--dry" in sys.argv:
        b = Bot(dry=True)
        b.scan()
    else:
        Bot().run()
