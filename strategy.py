"""
5-Pillar Quick-Limit Precision System v2.0 -- strategy engine.

Pure logic: no network, no Telegram. Takes candle DataFrames, returns a Setup
(or a reason string for why there is none) and renders the 13-point alert.

Candle DataFrames: UTC DatetimeIndex, columns open/high/low/close/volume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


# --------------------------------------------------------------------------
# Configuration (strategy constants are the rules from the system prompt)
# --------------------------------------------------------------------------
@dataclass
class Config:
    # account (bot refreshes these before every scan)
    capital_inr: float = 34733.0
    usdt_inr: float = 90.0

    # risk rules
    leverage: int = 5
    risk_pct_target: float = 0.0035      # aim inside the 0.31%-0.42% band
    risk_pct_min: float = 0.0031
    risk_pct_max: float = 0.0042
    risk_min_inr: float = 110.0
    risk_max_inr: float = 150.0
    fee_in: float = 0.0005               # per-side fee assumptions: VERIFY on CoinDCX
    fee_out: float = 0.0005

    # tranche architecture (multiples of 1H ATR)
    t1_atr_min: float = 1.2
    t1_atr_max: float = 1.5
    t2_atr: float = 2.2
    confluence_tol_atr: float = 0.15

    # take profits / stops
    tp1_pct: float = 1.25
    tp1_size: float = 0.70
    tp2_pct: float = 2.00                # allowed range 2.00-2.25
    sl_min_pct: float = 1.20
    sl_max_pct: float = 1.60
    sl_atr_buffer: float = 0.30
    breakeven_pct: float = 0.75
    time_stop_h: float = 6.0

    # session / macro gates (IST)
    blackout_start: tuple = (5, 0)
    blackout_end: tuple = (9, 30)
    friday_cutoff: tuple = (17, 30)      # no new setups Fri after this until Mon
    event_pre_h: float = 6.0             # stay flat this long before a high-impact event
    event_post_h: float = 1.0

    # filters / scoring
    exhaustion_ratio: float = 1.25       # last-24h range vs 14d average
    min_score: int = 70
    primary_score: int = 80

    # lot sizes (VERIFY against CoinDCX contract specs)
    qty_step: dict = field(default_factory=lambda: {"BTC": 0.001, "ETH": 0.01, "SOL": 0.1})
    default_qty_decimals: int = 3


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """Returns (line, direction) arrays. direction: +1 green, -1 red."""
    a = atr(df, period).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    close = df["close"].values
    n = len(df)
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    direction = np.ones(n)
    for i in range(1, n):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
        if close[i] > fub[i - 1]:
            direction[i] = 1
        elif close[i] < flb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    line = np.where(direction == 1, flb, fub)
    return line, direction


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def vwap_bands(df: pd.DataFrame, win: int = 24):
    """Rolling 24-bar (24H on 1H data) volume-weighted VWAP and sigma."""
    d = df.tail(win)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    v = d["volume"].replace(0, np.nan).fillna(1e-9)
    vw = float((tp * v).sum() / v.sum())
    sd = math.sqrt(float((((tp - vw) ** 2) * v).sum() / v.sum()))
    return vw, sd


def volume_profile(df: pd.DataFrame, bins: int = 60, va: float = 0.70) -> dict:
    """Fixed-range volume profile: vPOC, VAL, VAH, HVNs."""
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if hi <= lo:
        hi = lo * 1.0001
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol = np.zeros(bins)
    for h, l, v in zip(df["high"].values, df["low"].values, df["volume"].values):
        i0 = int(np.clip(np.searchsorted(edges, l, side="right") - 1, 0, bins - 1))
        i1 = int(np.clip(np.searchsorted(edges, h, side="right") - 1, 0, bins - 1))
        vol[i0:i1 + 1] += v / (i1 - i0 + 1)
    poc = int(vol.argmax())
    total, acc = vol.sum(), vol[poc]
    lo_i = hi_i = poc
    while acc < va * total and (lo_i > 0 or hi_i < bins - 1):
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        dn = vol[lo_i - 1] if lo_i > 0 else -1
        if up >= dn:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    mean = vol.mean()
    hvn = [float(centers[i]) for i in range(1, bins - 1)
           if vol[i] > vol[i - 1] and vol[i] >= vol[i + 1] and vol[i] > 1.3 * mean and i != poc]
    return dict(poc=float(centers[poc]), val=float(edges[lo_i]), vah=float(edges[hi_i + 1]),
                hvn=hvn, edges=edges, vol=vol)


def vol_percentile(profile: dict, price: float) -> float:
    i = int(np.clip(np.searchsorted(profile["edges"], price, side="right") - 1, 0, len(profile["vol"]) - 1))
    return float((profile["vol"] < profile["vol"][i]).mean())


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule, origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna()


def swing_points(df: pd.DataFrame, lookback: int = 48):
    d = df.tail(lookback)
    h, l = d["high"].values, d["low"].values
    highs = [float(h[i]) for i in range(2, len(d) - 2) if h[i] == h[i - 2:i + 3].max()]
    lows = [float(l[i]) for i in range(2, len(d) - 2) if l[i] == l[i - 2:i + 3].min()]
    return highs, lows


def higher_low_reversal(d15: pd.DataFrame, lookback: int = 48) -> bool:
    d = d15.tail(lookback)
    lows, highs = d["low"].values, d["high"].values
    n = len(d)
    sw = [i for i in range(2, n - 2) if lows[i] == lows[i - 2:i + 3].min()]
    if len(sw) < 2:
        return False
    i1, i2 = sw[-2], sw[-1]
    if lows[i2] <= lows[i1]:
        return False
    return float(d["close"].iloc[-1]) > float(highs[i1:i2 + 1].max())


# --------------------------------------------------------------------------
# Gates: session blackout, weekend, macro events
# --------------------------------------------------------------------------
@dataclass
class Event:
    name: str
    time: datetime          # tz-aware
    impact: str = "high"


def gate(now_ist: datetime, cfg: Config, events: list[Event]) -> str | None:
    m = now_ist.hour * 60 + now_ist.minute
    bs = cfg.blackout_start[0] * 60 + cfg.blackout_start[1]
    be = cfg.blackout_end[0] * 60 + cfg.blackout_end[1]
    if bs <= m < be:
        return "IST blackout window (05:00-09:30)"
    wd = now_ist.weekday()  # Mon=0
    if wd in (5, 6):
        return "weekend: no new setups Sat/Sun"
    if wd == 4 and m >= cfg.friday_cutoff[0] * 60 + cfg.friday_cutoff[1]:
        return "Friday cutoff: 6h time-stop would run into the weekend"
    for ev in events:
        if ev.impact != "high":
            continue
        if ev.time - timedelta(hours=cfg.event_pre_h) <= now_ist <= ev.time + timedelta(hours=cfg.event_post_h):
            return f"high-impact event inside hold window: {ev.name} ({ev.time.astimezone(IST):%a %d %b %H:%M IST})"
    return None


SESSIONS = [("Asia/Tokyo open", "Asia/Tokyo", 9, 0), ("London open", "Europe/London", 8, 0),
            ("New York open", "America/New_York", 9, 30)]


def next_sessions(now_ist: datetime, n: int = 3):
    out = []
    for name, tz, hh, mm in SESSIONS:
        z = ZoneInfo(tz)
        loc = now_ist.astimezone(z)
        cand = loc.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if cand <= loc:
            cand += timedelta(days=1)
        # skip weekend opens
        while cand.weekday() >= 5:
            cand += timedelta(days=1)
        out.append((name, cand.astimezone(IST)))
    out.sort(key=lambda x: x[1])
    return out[:n]


# --------------------------------------------------------------------------
# Setup object
# --------------------------------------------------------------------------
@dataclass
class Setup:
    symbol: str
    side: str                      # "long" | "short"
    path: str                      # "trend" | "reversal"
    price: float
    atr1h: float
    t1: float
    t2: float
    blended: float
    sl: float
    sl_pct: float
    sl_clamped: bool
    tp1: float
    tp2: float
    tp2_pct: float
    rr: float
    qty_t1: float
    qty_t2: float
    risk_inr: float
    risk_usdt: float
    risk_pct_capital: float
    margin_inr: float
    margin_usdt: float
    tp1_inr: float
    tp2_inr: float
    be_inr: float
    sources: list
    score: int
    score_parts: dict
    notes: dict
    created_ist: str
    rank: str = "Secondary"

    @property
    def coin(self) -> str:
        return self.symbol.replace("_USDT", "").replace("B-", "")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _round_down(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step


def _step(cfg: Config, coin: str) -> float:
    return cfg.qty_step.get(coin, 10 ** (-cfg.default_qty_decimals))


def evaluate(symbol, d15, d1h, d4h, refs, now_ist, cfg: Config, events: list[Event]):
    """Returns (Setup | None, reason)."""
    coin = symbol.replace("_USDT", "").replace("B-", "")
    if len(d1h) < 220 or len(d4h) < 55 or len(d15) < 60:
        return None, "insufficient candle history"

    price = float(d15["close"].iloc[-1])
    a1 = float(atr(d1h, 14).iloc[-1])
    if not a1 > 0:
        return None, "ATR unavailable"
    atr_pct = a1 / price * 100

    close1 = d1h["close"]
    e20, e50, e100, e200 = (float(ema(close1, n).iloc[-1]) for n in (20, 50, 100, 200))
    st_line, st_dir = supertrend(d1h, 10, 3.0)
    st_line, st_dir = float(st_line[-1]), int(st_dir[-1])
    bb_mid, bb_up, bb_lo = bollinger(close1)
    bb_up, bb_lo = float(bb_up.iloc[-1]), float(bb_lo.iloc[-1])

    # ---- exhaustion filter (session ATR) ----
    last24 = d1h.tail(24)
    rng_now = float(last24["high"].max() - last24["low"].min())
    prev = [d1h.iloc[-24 * (k + 2):-24 * (k + 1)] for k in range(13)]
    prev_r = [float(p["high"].max() - p["low"].min()) for p in prev if len(p) == 24]
    exh = rng_now / (sum(prev_r) / len(prev_r)) if prev_r else 1.0
    if exh >= cfg.exhaustion_ratio:
        return None, f"session ATR exhausted (24h range {exh:.2f}x 14d avg)"

    # ---- directional rules ----
    side = path = None
    if st_dir == -1 and price < st_line and price < e50:
        side, path = "short", "trend"
    elif st_dir == 1 and price > st_line:
        side, path = "long", "trend"
    elif higher_low_reversal(d15):
        side, path = "long", "reversal"
    if side is None:
        return None, "no directional alignment (1H SuperTrend / 50 EMA / 15M higher-low)"
    sgn = 1 if side == "short" else -1   # entries sit above price for shorts, below for longs

    # ---- level map ----
    prof1 = volume_profile(d1h.tail(72))
    prof4 = volume_profile(d4h.tail(60))
    vw, vsd = vwap_bands(d1h, 24)
    levels = [("1H BB " + ("upper" if side == "short" else "lower"), bb_up if side == "short" else bb_lo, "ind"),
              ("1H EMA20", e20, "ind"), ("1H EMA50", e50, "ind"), ("1H EMA100", e100, "ind"), ("1H EMA200", e200, "ind"),
              ("1H VAH", prof1["vah"], "struct"), ("1H vPOC", prof1["poc"], "struct"), ("1H VAL", prof1["val"], "struct"),
              ("4H VAH", prof4["vah"], "struct"), ("4H vPOC", prof4["poc"], "struct"), ("4H VAL", prof4["val"], "struct"),
              ("VWAP +2s", vw + 2 * vsd, "struct"), ("VWAP -2s", vw - 2 * vsd, "struct"),
              ("VWAP +3s", vw + 3 * vsd, "struct"), ("VWAP -3s", vw - 3 * vsd, "struct")]
    levels += [("1H HVN", p, "struct") for p in prof1["hvn"]]
    levels += [("4H HVN", p, "struct") for p in prof4["hvn"]]

    def dist(p):
        return sgn * (p - price) / a1

    band = [(n, p, k) for n, p, k in levels if cfg.t1_atr_min <= dist(p) <= cfg.t1_atr_max]
    best = None
    for n0, p0, k0 in band:
        grp = [x for x in band if abs(x[1] - p0) <= cfg.confluence_tol_atr * a1]
        kinds = {x[2] for x in grp}
        if kinds != {"ind", "struct"}:
            continue
        md = float(np.mean([dist(x[1]) for x in grp]))
        key = (len(grp), -abs(md - 1.35))
        if best is None or key > best[0]:
            best = (key, grp)
    if best is None:
        return None, f"no BB/EMA + FRVP/VWAP confluence inside {cfg.t1_atr_min}-{cfg.t1_atr_max}x 1H ATR"
    grp = best[1]
    t1 = float(np.mean([x[1] for x in grp]))
    lo_px, hi_px = sorted([price + sgn * cfg.t1_atr_min * a1, price + sgn * cfg.t1_atr_max * a1])
    t1 = min(max(t1, lo_px), hi_px)
    t2 = price + sgn * cfg.t2_atr * a1
    blended = (t1 + t2) / 2

    # ---- stop loss behind a 1H structural wick + ATR buffer ----
    sw_hi, sw_lo = swing_points(d1h, 48)
    if side == "short":
        cands = [h for h in sw_hi if t2 - 0.25 * a1 <= h <= t2 + 1.0 * a1]
        wick = max(cands) if cands else t2
        sl_raw = max(wick, t2) + cfg.sl_atr_buffer * a1
    else:
        cands = [l for l in sw_lo if t2 - 1.0 * a1 <= l <= t2 + 0.25 * a1]
        wick = min(cands) if cands else t2
        sl_raw = min(wick, t2) - cfg.sl_atr_buffer * a1
    wick_found = bool(cands)
    raw_pct = abs(sl_raw - blended) / blended * 100
    clamped = False
    sl = sl_raw
    if raw_pct < cfg.sl_min_pct:
        sl = blended + sgn * blended * cfg.sl_min_pct / 100
        clamped = True
    elif raw_pct > cfg.sl_max_pct:
        sl = blended + sgn * blended * cfg.sl_max_pct / 100
        clamped = True
        # clamped stop must still sit behind T2 with a real cushion
        if sgn * (sl - t2) < 0.10 * a1:
            return None, (f"ATR too wide: 1.60% SL cap leaves <0.10 ATR cushion behind T2 "
                          f"(raw SL {raw_pct:.2f}% from blended)")
    if sgn * (sl - t2) <= 0:
        return None, "stop would sit inside T2"
    sl_pct = abs(sl - blended) / blended * 100

    # ---- sizing: risk inside 0.31%-0.42% / Rs110-150, fees included ----
    cap = cfg.capital_inr
    lo = max(cfg.risk_min_inr, cap * cfg.risk_pct_min)
    hi = min(cfg.risk_max_inr, cap * cfg.risk_pct_max)
    lo = min(lo, hi)
    risk_target = min(max(cap * cfg.risk_pct_target, lo), hi)
    cost_frac = sl_pct / 100 + cfg.fee_in + cfg.fee_out
    notional_inr = risk_target / cost_frac
    q_total = notional_inr / cfg.usdt_inr / blended
    step = _step(cfg, coin)
    q_t = _round_down(q_total / 2, step)
    if q_t <= 0:
        return None, f"tranche size rounds below minimum lot ({step} {coin}); risk cap too small for this pair"
    usd = cfg.usdt_inr
    price_loss_usdt = q_t * abs(sl - t1) + q_t * abs(sl - t2)
    fees_usdt = cfg.fee_in * q_t * (t1 + t2) + cfg.fee_out * q_t * 2 * sl
    risk_usdt = price_loss_usdt + fees_usdt
    risk_inr = risk_usdt * usd
    if risk_inr < 0.6 * risk_target:
        return None, f"lot step too coarse: rounded risk Rs{risk_inr:.0f} < 60% of target Rs{risk_target:.0f}"
    if risk_inr > min(cfg.risk_max_inr, cap * cfg.risk_pct_max) + 1e-6:
        return None, "rounded risk exceeds hard cap"
    q_sum = 2 * q_t
    notional_usdt = q_t * (t1 + t2)
    margin_usdt = notional_usdt / cfg.leverage
    margin_inr = margin_usdt * usd

    # ---- targets ----
    tp1 = blended * (1 - sgn * cfg.tp1_pct / 100)
    tp2 = blended * (1 - sgn * cfg.tp2_pct / 100)
    tp1_inr = cfg.tp1_size * q_sum * blended * cfg.tp1_pct / 100 * usd
    tp2_inr = (1 - cfg.tp1_size) * q_sum * blended * cfg.tp2_pct / 100 * usd
    blended_exit_pct = cfg.tp1_size * cfg.tp1_pct + (1 - cfg.tp1_size) * cfg.tp2_pct
    rr = blended_exit_pct / sl_pct
    be_inr = q_sum * blended * cfg.breakeven_pct / 100 * usd

    # ---- context for scoring ----
    # 4H trend
    c4 = d4h["close"]
    e50_4 = float(ema(c4, 50).iloc[-1])
    _, dir4 = supertrend(d4h, 10, 3.0)
    want = -1 if side == "short" else 1
    t4 = (1 if (float(c4.iloc[-1]) - e50_4) * want > 0 else 0) + (1 if int(dir4[-1]) == want else 0)
    # reference asset alignment (BTC for alts/ETH, ETH for BTC)
    ref_key = "ETH" if coin == "BTC" else "BTC"
    ref_ok, smt = False, "n/a"
    if ref_key in refs and len(refs[ref_key]) > 60:
        _, rdir = supertrend(refs[ref_key], 10, 3.0)
        ref_ok = int(rdir[-1]) == want
        smt = smt_note(d1h, refs[ref_key], coin, ref_key)
    ev_near = [e for e in events if e.impact == "high" and 0 <= (e.time - now_ist).total_seconds() <= 24 * 3600]

    # prior sweeps of T2 in last 24h
    last = d1h.tail(24)
    swept = int(((last["high"] >= t2).sum() if side == "short" else (last["low"] <= t2).sum()))
    vp = vol_percentile(prof1, t1)
    n_struct = sum(1 for x in grp if x[2] == "struct")
    n_ind = sum(1 for x in grp if x[2] == "ind")

    A = min(10, 5 * n_struct) + min(8, 4 * n_ind) + 7 * vp
    gap = abs(sl - t2) / a1
    B = (6 if wick_found else 3) + min(8, 8 * gap / 0.5) + max(0, 6 - 2 * swept) + (5 if not clamped else 2)
    C = ((8 if path == "trend" else 3) + (4 if sgn * (e50 - price) > 0 else 1) + {0: 0, 1: 3, 2: 6}[t4]
         + (4 if ref_ok else 1) + (3 if not ev_near else 1))
    exh_pts = 4 if exh <= 0.9 else max(0.0, 4 * (cfg.exhaustion_ratio - exh) / (cfg.exhaustion_ratio - 0.9))
    reach = atr_pct * math.sqrt(cfg.time_stop_h) / cfg.tp1_pct
    D = 6 + 5 + min(6, 6 * rr / 1.2) + exh_pts + 4 * min(1.0, reach)
    parts = {"A": round(min(25, A)), "B": round(min(25, B)), "C": round(min(25, C)), "D": round(min(25, D))}
    score = int(sum(parts.values()))

    notes = dict(
        e20=e20, e50=e50, e200=e200, st_line=st_line, st_dir=st_dir, bb_up=bb_up, bb_lo=bb_lo,
        vah=prof1["vah"], val=prof1["val"], poc=prof1["poc"], vwap=vw, vsd=vsd, atr_pct=atr_pct,
        exh=exh, t4=t4, ref_key=ref_key, ref_ok=ref_ok, smt=smt, wick=wick, wick_found=wick_found,
        swept=swept, vp=vp, gap=gap, reach=reach, raw_sl_pct=raw_pct, risk_target=risk_target,
        notional_usdt=notional_usdt, price_loss_usdt=price_loss_usdt, fees_usdt=fees_usdt,
        blended_exit_pct=blended_exit_pct, rr=rr,
        d15_trend=("higher-low reversal confirmed" if path == "reversal" else
                   ("15M below 1H structure" if side == "short" else "15M above 1H structure")),
        events_24h=[f"{e.name} {e.time.astimezone(IST):%a %H:%M IST}" for e in ev_near],
    )
    setup = Setup(
        symbol=symbol, side=side, path=path, price=price, atr1h=a1, t1=t1, t2=t2, blended=blended, sl=sl,
        sl_pct=sl_pct, sl_clamped=clamped, tp1=tp1, tp2=tp2, tp2_pct=cfg.tp2_pct, rr=rr,
        qty_t1=q_t, qty_t2=q_t, risk_inr=risk_inr, risk_usdt=risk_usdt, risk_pct_capital=risk_inr / cap * 100,
        margin_inr=margin_inr, margin_usdt=margin_usdt, tp1_inr=tp1_inr, tp2_inr=tp2_inr, be_inr=be_inr,
        sources=[(n, p, k) for n, p, k in grp], score=score, score_parts=parts, notes=notes,
        created_ist=now_ist.strftime("%Y-%m-%d %H:%M IST"),
    )
    if score < cfg.min_score:
        return None, f"score {score} below minimum {cfg.min_score}"
    return setup, "ok"


def smt_note(d1h: pd.DataFrame, ref: pd.DataFrame, coin: str, ref_key: str, win: int = 24, recent: int = 6) -> str:
    a, b = d1h.tail(win), ref.tail(win)
    if len(a) < win or len(b) < win:
        return "n/a"
    a_lo, b_lo = a["low"].iloc[-recent:].min() <= a["low"].iloc[:-recent].min(), b["low"].iloc[-recent:].min() <= b["low"].iloc[:-recent].min()
    a_hi, b_hi = a["high"].iloc[-recent:].max() >= a["high"].iloc[:-recent].max(), b["high"].iloc[-recent:].max() >= b["high"].iloc[:-recent].max()
    if a_lo != b_lo:
        who = coin if a_lo else ref_key
        return f"bullish SMT: only {who} swept the 24H low"
    if a_hi != b_hi:
        who = coin if a_hi else ref_key
        return f"bearish SMT: only {who} swept the 24H high"
    return "no SMT divergence"


def rank_setups(setups: list[Setup]) -> list[Setup]:
    setups = sorted(setups, key=lambda s: s.score, reverse=True)
    for i, s in enumerate(setups):
        s.rank = "Primary" if i == 0 else "Secondary"
    return setups


# --------------------------------------------------------------------------
# Rendering (13-point template, zero deviation)
# --------------------------------------------------------------------------
def fpx(p: float) -> str:
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 10:
        return f"{p:.3f}"
    if p >= 1:
        return f"{p:.4f}"
    return f"{p:.5f}"


def inr(x: float, sign: bool = False) -> str:
    s = f"{abs(x):,.2f}"
    if sign:
        return ("+" if x >= 0 else "-") + "₹" + s
    return "₹" + s


def usd(x: float, sign: bool = False) -> str:
    s = f"{abs(x):,.2f}"
    if sign:
        return ("+" if x >= 0 else "-") + "$" + s
    return "$" + s


def dashboard(st: dict, status: str) -> str:
    ini, cur, tgt = st["initial_inr"], st["current_inr"], st["target_inr"]
    r = st["usdt_inr"]
    pnl = cur - ini
    peak = max(st.get("peak_inr", ini), cur)
    dd = peak - cur
    prog = (cur - ini) / (tgt - ini) * 100 if tgt > ini else 0.0
    return "\n".join([
        "* SESSION P&L & RECOVERY DASHBOARD:",
        f"  * Initial Starting Capital: {usd(ini / r)} / {inr(ini)}",
        f"  * Current Account Value: {usd(cur / r)} / {inr(cur)}",
        f"  * Cumulative Realized P&L: {usd(pnl / r, True)} / {inr(pnl, True)} ({pnl / ini * 100:+.2f}% Account Growth)",
        f"  * Active Drawdown: -{usd(dd / r)[0:]} / -{inr(dd)} (-{dd / peak * 100:.2f}%)",
        f"  * Target Goal Milestone: {tgt / ini:.2f}x Target Valuation: {usd(tgt / r)} / {inr(tgt)}",
        f"  * Progress Toward Target: Target Completion: {prog:.2f}% | Remaining: {usd(max(tgt - cur, 0) / r)} / {inr(max(tgt - cur, 0))}",
        f"  * Active Execution Status: {status}",
    ])


def render_alert(s: Setup, trade_id: str, cfg: Config, st: dict, now_ist: datetime, events: list[Event]) -> str:
    n = s.notes
    long_ = s.side == "long"
    word = "Limit Buy" if long_ else "Limit Sell"
    verb = "BUY" if long_ else "SELL"
    exit_verb = "SELL" if long_ else "BUY"
    r = cfg.usdt_inr
    q = s.qty_t1 + s.qty_t2
    coin = s.coin
    step_dec = max(0, -int(math.floor(math.log10(_step(cfg, coin)))))
    fq = lambda x: f"{x:.{step_dec}f}"

    # session audit text
    sess = next_sessions(now_ist)
    sess_txt = "; ".join(f"{nm} {t:%a %H:%M} IST (in {int((t - now_ist).total_seconds() // 3600)}h{int((t - now_ist).total_seconds() % 3600 // 60):02d}m)"
                         for nm, t in sess)
    within = [nm for nm, t in sess if (t - now_ist).total_seconds() <= cfg.time_stop_h * 3600]
    manip = (f"AMD sweep risk INSIDE the 6h hold window ({', '.join(within)}); levels already sized on 1H ATR, SL kept behind wick + ATR buffer."
             if within else "No session open inside the 6h hold window; standard AMD wick risk only.")
    ev_txt = ("High-impact events next 24h: " + "; ".join(n["events_24h"]) + ". (Outside the 6h hold window, otherwise bot would have stayed flat.)"
              if n["events_24h"] else "No high-impact event in the bot's calendar for the next 24h (calendar is manual: keep events.json current).")

    # multi-timeframe audit
    stc = "green" if n["st_dir"] == 1 else "red"
    audit = [
        "MULTI-TIMEFRAME AUDIT",
        f"* 4H: trend score {n['t4']}/2 vs {s.side} bias (4H close vs 50 EMA + 4H SuperTrend); {n['ref_key']} 1H SuperTrend {'aligned' if n['ref_ok'] else 'NOT aligned'}",
        f"* 1H: SuperTrend {stc} @ {fpx(n['st_line'])}, 50 EMA {fpx(n['e50'])}, ATR {fpx(s.atr1h)} ({n['atr_pct']:.2f}%), BB {fpx(n['bb_lo'])}-{fpx(n['bb_up'])}, VAL/vPOC/VAH {fpx(n['val'])}/{fpx(n['poc'])}/{fpx(n['vah'])}",
        f"* 15M: {n['d15_trend']}; price {fpx(s.price)}",
    ]
    srcs = ", ".join(f"{nm} {fpx(p)}" for nm, p, _ in s.sources)

    # math proof
    lossT1 = s.qty_t1 * abs(s.sl - s.t1)
    lossT2 = s.qty_t2 * abs(s.sl - s.t2)
    proof = [
        "MATHEMATICAL PROOF",
        f"* T1 = price {'-' if long_ else '+'} {abs(s.t1 - s.price) / s.atr1h:.2f}x 1H ATR; T2 = price {'-' if long_ else '+'} {abs(s.t2 - s.price) / s.atr1h:.2f}x 1H ATR; blended = ({fpx(s.t1)} + {fpx(s.t2)}) / 2 = {fpx(s.blended)}",
        f"* SL raw distance {n['raw_sl_pct']:.2f}% -> {'clamped to ' if s.sl_clamped else 'kept at '}{s.sl_pct:.2f}% (rule 1.20%-1.60%); SL is {abs(s.sl - s.t2) / s.atr1h:.2f} ATR beyond T2",
        f"* Risk if both fill and SL hits: T1 leg {usd(lossT1)} + T2 leg {usd(lossT2)} + est. fees {usd(n['fees_usdt'])} = {usd(s.risk_usdt)} x {r:.2f} = {inr(s.risk_inr)} ({s.risk_pct_capital:.2f}% of {inr(cfg.capital_inr)})",
        f"* TP1: 70% x {fq(q)} x {fpx(s.blended)} x 1.25% = {inr(s.tp1_inr)}; TP2: 30% x 2.00% = {inr(s.tp2_inr)}; blended exit {n['blended_exit_pct']:.3f}% / SL {s.sl_pct:.2f}% = {s.rr:.2f}R (gross)",
        f"* Margin: notional {usd(n['notional_usdt'])} / {cfg.leverage}x = {usd(s.margin_usdt)} = {inr(s.margin_inr)} ({s.margin_inr / cfg.capital_inr * 100:.1f}% of account)",
    ]
    orders = [
        "ACTION DIRECTIVES (CoinDCX Futures)",
        f"1. Open {coin}/USDT futures. Set margin mode ISOLATED, leverage {cfg.leverage}x.",
        f"2. VERIFY FIRST (bot cannot): liquidation heatmap + CVD absorption near T1. SMT check: {n['smt']}. Skip if they contradict.",
        f"3. Tranche 1: {word} {fq(s.qty_t1)} {coin} @ {fpx(s.t1)}.",
        f"4. Tranche 2: {word} {fq(s.qty_t2)} {coin} @ {fpx(s.t2)}.",
        f"5. Stop-loss on the full {fq(q)} {coin}: trigger {fpx(s.sl)} (reduce-only).",
        f"6. TP1: {exit_verb} limit {fq(q * cfg.tp1_size)} @ {fpx(s.tp1)}. TP2: {exit_verb} limit {fq(q * (1 - cfg.tp1_size))} @ {fpx(s.tp2)}.",
        f"7. Send me your staging screenshot, then reply /staged {trade_id}. I will verify the numbers and start tracking.",
        f"8. In trade: at +0.75% (~{inr(s.be_inr)}) move SL to breakeven + fees. 6h after fill with no TP1: close at breakeven/minor profit. If price reclaims structure after T1 without touching T2: cancel T2.",
    ]

    f = []
    f.append(f"ASSET / PAIR: {coin}/USDT")
    f.append(f"CURRENT ASSET PRICE: {fpx(s.price)}")
    f.append(f"SETUP RANKING & SCORE: {s.rank} Setup | Score: {s.score}/100" + ("  [15M reversal path]" if s.path == "reversal" else ""))
    f.append(f"DIRECTION & TYPE: Dual-Tranche {word}")
    f.append(f"LEVERAGE: Isolated {cfg.leverage}x")
    f.append(f"ENTRY ZONE / LIMIT: Tranche 1: {fpx(s.t1)} | Tranche 2: {fpx(s.t2)} | Blended Entry: {fpx(s.blended)}")
    f.append(f"STOP LOSS (SL): {fpx(s.sl)} (Distance: {s.sl_pct:.2f}% | Max Loss: -{usd(s.risk_usdt)} USDT / ~{inr(s.risk_inr)} INR | {s.risk_pct_capital:.2f}% Portfolio Risk)")
    f.append(f"TAKE PROFIT (TP): TP1 (70%): {fpx(s.tp1)} (+1.25% | ~{inr(s.tp1_inr)} INR) | TP2 (30%): {fpx(s.tp2)} (+{s.tp2_pct:.2f}% | ~{inr(s.tp2_inr)} INR)")
    f.append(f"RISK-TO-REWARD (R:R): {s.rr:.2f} R based on TP1/TP2 Blended Exit")
    f.append(f"POSITION SIZE & MARGIN: Total {fq(q)} {coin} (T1: 50% = {fq(s.qty_t1)} | T2: 50% = {fq(s.qty_t2)}) (~{usd(s.margin_usdt)} USDT / ~{inr(s.margin_inr)} INR required margin)")
    f.append(f"UPCOMING SESSION & VOLATILITY AUDIT: Next opens: {sess_txt}. {manip} {ev_txt} 24h range is {n['exh']:.2f}x the 14-day average (limit 1.25x).")
    p = s.score_parts
    f.append("STRUCTURAL RATIONALE & SCORE BREAKDOWN:")
    f.append(f"  * Order Book Liquidity & Volume ({p['A']}/25): T1 at {fpx(s.t1)} sits {abs(s.t1 - s.price) / s.atr1h:.2f}x 1H ATR away with confluence of [{srcs}]; volume at that price ranks at percentile {n['vp'] * 100:.0f} of the 3-day profile.")
    f.append(f"  * Sweep Immunity ({p['B']}/25): T2 at {abs(s.t2 - s.price) / s.atr1h:.2f}x 1H ATR; {'real 1H swing wick at ' + fpx(n['wick']) if n['wick_found'] else 'no swing wick near T2 so T2 itself is the anchor'}; SL {fpx(s.sl)} = {n['gap']:.2f} ATR beyond T2 (0.30 ATR buffer); T2 level was reached {n['swept']}x in the last 24 1H bars.")
    f.append(f"  * Macro Catalyst & Directional Trend ({p['C']}/25): 1H SuperTrend {stc}, price {'below' if not long_ else 'above'} 1H 50 EMA = {'yes' if (s.price < n['e50']) == (not long_) else 'no'}; 4H alignment {n['t4']}/2; {n['ref_key']} {'aligned' if n['ref_ok'] else 'not aligned'}; {'no high-impact event inside 24h' if not n['events_24h'] else 'event inside 24h but outside hold window'}.")
    f.append(f"  * Capital Preservation ({p['D']}/25): risk {inr(s.risk_inr)} = {s.risk_pct_capital:.2f}% of account (cap -Rs110 to -Rs150 / 0.31%-0.42%); SL {s.sl_pct:.2f}% inside 1.20%-1.60%; 6h time-stop; TP1 reach factor {min(1, n['reach']):.2f} (1H ATR {n['atr_pct']:.2f}%).")

    head = f"SETUP ALERT {trade_id} | {s.created_ist}"
    dash = dashboard(st, "Alert issued - awaiting your staging screenshot")
    body = _render_fields(f)
    return "\n\n".join([head, "\n".join(audit), body, "\n".join(proof), "\n".join(orders), dash])


def _render_fields(f: list[str]) -> str:
    out, i = [], 0
    for line in f:
        if line.startswith("  "):
            out.append(line)
        else:
            i += 1
            out.append(f"* {line}")
    return "\n".join(out)
