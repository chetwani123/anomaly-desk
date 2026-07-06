#!/usr/bin/env python3
"""
anomaly_desk.py — daily five-strategy screener, delivered to your inbox.

ARCHITECTURE (why it's split this way):
  Strategies 1, 2, 5 (trend / rotation / mean-reversion) are pure price math.
    -> computed EXACTLY from yfinance daily bars. No LLM, no approximation.
  Strategies 3, 4 (quality / PEAD) require reading news + fundamentals.
    -> delegated to Claude with web search. LLM is used only where research
       is genuinely needed, never for arithmetic.

RUN:      python anomaly_desk.py            (prints report, emails if SMTP env set)
SCHEDULE: GitHub Actions cron (see anomaly_desk.yml) or local crontab:
          30 13 * * 1-5  cd ~/desk && ./venv/bin/python anomaly_desk.py
          (13:30 UTC = 6:30am PT, pre-market)

ENV VARS:
  ANTHROPIC_API_KEY   required for strategies 3-4 (script degrades gracefully without)
  DESK_EMAIL_TO       optional — where to send the report
  DESK_EMAIL_FROM     optional — gmail address to send from
  DESK_EMAIL_PASS     optional — gmail APP password (not your real password;
                      create at myaccount.google.com/apppasswords)
"""

import os
import smtplib
import sys
from datetime import datetime
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------
# CONFIG — the only section you should need to edit
# ----------------------------------------------------------------------------

TREND_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "DBC"]

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY",
               "XLP", "XLU", "XLB", "XLRE", "XLC"]

# Mean-reversion universe: liquid ETFs + megacaps. Liquidity matters because
# the edge is liquidity PROVISION — thin names don't pay you for panic.
MEANREV_UNIVERSE = ["SPY", "QQQ", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY",
                    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO",
                    "SOXX", "SMH"]

MOM_LOOKBACK_D = 126     # ~6 months of trading days
MOM_SKIP_D = 10          # skip most recent 2 weeks (short-term reversal effect)
MOM_TOP_N = 4

RSI2_TRIGGER = 10.0      # Connors threshold: RSI(2) below this = oversold
MEANREV_MAX_PICKS = 4

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# ----------------------------------------------------------------------------
# DATA — one batched download for everything price-based
# ----------------------------------------------------------------------------

def fetch_closes(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """One yfinance call for all tickers -> DataFrame of adjusted closes.
    Batched because N separate calls is how you get rate-limited."""
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, group_by="column")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    # Drop tickers that came back empty (delisted / bad symbol) rather than crash.
    return closes.dropna(axis=1, how="all")


# ----------------------------------------------------------------------------
# SIGNAL MATH — each function is one strategy's exact rule
# ----------------------------------------------------------------------------

def rsi(series: pd.Series, period: int) -> float:
    """Wilder's RSI. period=2 for the Connors mean-reversion trigger.
    Wilder smoothing = EMA with alpha 1/period (NOT the common 2/(n+1))."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)   # avoid 0-division on pure uptrends
    out = 100 - 100 / (1 + rs)
    return float(out.iloc[-1].item()) if not np.isnan(out.iloc[-1]) else 100.0


def consecutive_down_days(series: pd.Series) -> int:
    """Count of consecutive closes lower than the prior close, ending today."""
    diffs = series.diff().dropna()
    n = 0
    for d in reversed(diffs.to_list()):
        if d < 0:
            n += 1
        else:
            break
    return n


def trend_screen(closes: pd.DataFrame) -> dict:
    """Strategy 1 — time-series momentum (Faber 200DMA rule).
    Signal: price vs 200DMA. Regime: fraction of universe above trend."""
    rows = []
    for t in TREND_UNIVERSE:
        if t not in closes.columns:
            continue
        s = closes[t].dropna()
        if len(s) < 210:                      # need a full 200DMA + slope window
            continue
        ma200 = s.rolling(200).mean().iloc[-1]
        px = s.iloc[-1]
        pct_above = (px / ma200 - 1) * 100
        # 3-month slope of the 200DMA itself: is the trend line rising?
        ma_series = s.rolling(200).mean()
        slope_up = ma_series.iloc[-1] > ma_series.iloc[-63]
        rows.append({"ticker": t, "px": px, "pct_above": pct_above,
                     "above": px > ma200, "ma_rising": slope_up})
    rows.sort(key=lambda r: r["pct_above"], reverse=True)
    n_above = sum(r["above"] for r in rows)
    regime = ("RISK-ON" if n_above >= 6 else
              "MIXED" if n_above >= 3 else "RISK-OFF")
    return {"regime": f"{regime} ({n_above}/{len(rows)} above 200DMA)",
            "picks": [r for r in rows if r["above"]][:4],
            "all": rows}


def rotation_screen(closes: pd.DataFrame) -> dict:
    """Strategy 2 — cross-sectional momentum (Jegadeesh-Titman).
    Rank sectors by 6-month return, SKIPPING the last 2 weeks: short-horizon
    returns mean-revert, so including them degrades the momentum signal."""
    rows = []
    for t in SECTOR_ETFS:
        if t not in closes.columns:
            continue
        s = closes[t].dropna()
        if len(s) < MOM_LOOKBACK_D + MOM_SKIP_D + 1:
            continue
        ret = (s.iloc[-1 - MOM_SKIP_D] / s.iloc[-1 - MOM_SKIP_D - MOM_LOOKBACK_D] - 1) * 100
        recent = (s.iloc[-1] / s.iloc[-1 - MOM_SKIP_D] - 1) * 100   # shown, not ranked
        rows.append({"ticker": t, "mom_6m": ret, "recent_2w": recent})
    rows.sort(key=lambda r: r["mom_6m"], reverse=True)
    # Rotation-instability flag: is a top-3 name bleeding right now?
    unstable = any(r["recent_2w"] < -3.0 for r in rows[:3])
    return {"picks": rows[:MOM_TOP_N], "unstable": unstable, "all": rows}


def meanrev_screen(closes: pd.DataFrame) -> dict:
    """Strategy 5 — Connors-style pullback buying, ONLY above the 200DMA.
    The trend filter is the strategy: below trend, 'oversold' is a falling knife."""
    setups, near = [], []
    for t in MEANREV_UNIVERSE:
        if t not in closes.columns:
            continue
        s = closes[t].dropna()
        if len(s) < 210:
            continue
        ma200 = s.rolling(200).mean().iloc[-1]
        if s.iloc[-1] <= ma200:
            continue                                   # trend filter: hard gate
        r2 = rsi(s, 2)
        down = consecutive_down_days(s)
        pullback = (s.iloc[-1] / s.iloc[-10:].max() - 1) * 100  # % off 10d high
        row = {"ticker": t, "rsi2": r2, "down_days": down,
               "off_10d_high": pullback,
               "pct_above_200": (s.iloc[-1] / ma200 - 1) * 100}
        if r2 < RSI2_TRIGGER or down >= 3:
            setups.append(row)
        elif r2 < 25 or down == 2:
            near.append(row)                           # watchlist, not triggered
    setups.sort(key=lambda r: r["rsi2"])
    near.sort(key=lambda r: r["rsi2"])
    return {"picks": setups[:MEANREV_MAX_PICKS], "near": near[:3]}


# ----------------------------------------------------------------------------
# LLM RESEARCH LEGS — quality + PEAD (the two that need reading, not math)
# ----------------------------------------------------------------------------

def claude_research(prompt: str) -> str:
    """One Claude call with web search enabled. Returns markdown text.
    Import is local so the price-math strategies work without the SDK installed."""
    import anthropic
    client = anthropic.Anthropic()                      # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        tools=[{"type": "web_search_20250305", "name": "web_search",
                "max_uses": 6}],
        messages=[{"role": "user", "content": prompt}],
    )
    return "\n".join(b.text for b in msg.content if b.type == "text").strip()


QUALITY_DEEP = f"""Today is {datetime.now():%A %B %d, %Y}.
You are a quality-compounder screener (Novy-Marx profitability, AQR QMJ, with a
duration-mispricing lens). Search current data, then apply THREE GATES in order:
GATE 1 (machine): ROIC > 15% for 5+ consecutive years; gross margins stable or
  EXPANDING (weight margin TRAJECTORY over level — 2+ quarters of unexplained GM
  slippage disqualifies); FCF conversion > 80% of net income; net debt/EBITDA < 2.
GATE 2 (runway): a credible place to reinvest at high returns — fragmented market,
  proven redeploy playbook (geographies, tuck-ins, adjacencies). High ROIC with no
  runway is a bond, not a compounder.
GATE 3 (price): FCF yield > 3.5% OR EV/EBIT below the company's own 5-yr median.
TILTS: prefer boring low-coverage names (<15 analysts: distributors, niche
  industrials, vertical software, specialty chemicals) over consensus mega-caps —
  the duration mispricing is largest where attention is thinnest. Include one
  serial-acquirer candidate if any qualifies (many small deals at 4-6x EBIT,
  flat/falling share count, founder or >5% insider ownership).
DISQUALIFIERS: single customer >20% of revenue; margins from a temporary pricing
  umbrella; adjusted-EBITDA persistently diverging from FCF; acquirer deal sizes
  creeping up.
Output 3 names: ticker | which gate evidence is strongest (one current number
each) | one-sentence duration thesis | the specific invalidator. Only numbers you
actually found; if a gate is unverified from search, SAY unverified rather than
assert. Under 200 words, plain text."""

QUALITY_CHECK = f"""Today is {datetime.now():%A %B %d, %Y}.
Fast moat-integrity check. For each ticker in this watchlist: {{watchlist}} —
search for its MOST RECENT quarterly results and answer pass/fail on two vitals:
(1) gross margin held or expanded vs year-ago, (2) no new disqualifier (customer
concentration, leverage jump, adjusted-vs-FCF divergence). One line per name:
TICKER: PASS/FAIL/WATCH — five-word reason. Flag any name reporting earnings in
the next 2 weeks. Under 100 words."""

# Deep screen on the first weekday of Jan/Apr/Jul/Oct (post-earnings-season) and
# every Monday; fast integrity check other days IF a watchlist is configured.
QUALITY_WATCHLIST: list[str] = []          # e.g. ["POOL", "WSO", "CPRT", "HEI"]


def quality_prompt() -> str:
    """Quarterly-deep / weekly-deep / daily-shallow cadence selector."""
    today = datetime.now()
    deep = today.weekday() == 0 or (today.month in (1, 4, 7, 10) and today.day <= 7)
    if deep or not QUALITY_WATCHLIST:
        return QUALITY_DEEP
    return QUALITY_CHECK.format(watchlist=", ".join(QUALITY_WATCHLIST))


PEAD_PROMPT = f"""Today is {datetime.now():%A %B %d, %Y}.
You are a post-earnings-announcement-drift screener. Search for companies that
reported in the LAST 3 WEEKS with ALL of: EPS beat + revenue beat + raised guidance
+ positive stock reaction on announcement day. Prefer mid-caps (thin coverage =
drift persists). Name up to 3 with: ticker, report date, the surprise numbers,
day-1 move, weeks of drift window remaining. If the window is thin (between
seasons), say so and give the best near-qualifiers. Under 150 words, plain text."""


# ----------------------------------------------------------------------------
# REPORT
# ----------------------------------------------------------------------------

def fmt_trend(res: dict) -> str:
    lines = [f"REGIME: {res['regime']}"]
    for r in res["all"]:
        flag = "LONG " if r["above"] else "cash "
        lines.append(f"  {flag} {r['ticker']:<5} {r['pct_above']:+6.1f}% vs 200DMA"
                     f"   MA {'rising' if r['ma_rising'] else 'falling'}")
    return "\n".join(lines)


def fmt_rotation(res: dict) -> str:
    lines = [f"TOP {MOM_TOP_N} (6mo return, ex last 2wk)"
             + ("   ⚠ leadership unstable" if res["unstable"] else "")]
    for i, r in enumerate(res["all"], 1):
        mark = "→" if i <= MOM_TOP_N else " "
        lines.append(f"  {mark} #{i:<2} {r['ticker']:<5} {r['mom_6m']:+6.1f}%"
                     f"   (2wk: {r['recent_2w']:+5.1f}%)")
    return "\n".join(lines)


def fmt_meanrev(res: dict) -> str:
    if not res["picks"]:
        lines = ["No triggered setups today (that's the filter working)."]
    else:
        lines = ["TRIGGERED (buy zone; exit: close > 5d MA or 5 sessions):"]
        for r in res["picks"]:
            lines.append(f"  → {r['ticker']:<5} RSI(2) {r['rsi2']:5.1f}"
                         f"   {r['down_days']}d down"
                         f"   {r['off_10d_high']:+5.1f}% off 10d high"
                         f"   ({r['pct_above_200']:+.1f}% above 200DMA)")
    if res["near"]:
        lines.append("  near-setups (not triggered): "
                     + ", ".join(f"{r['ticker']} RSI2={r['rsi2']:.0f}" for r in res["near"]))
    return "\n".join(lines)


def build_report() -> tuple[str, list[str]]:
    """Returns (report_text, warnings). warnings is non-empty when a research
    leg failed to complete, so callers can flag a degraded run up front."""
    warnings: list[str] = []
    closes = fetch_closes(sorted(set(TREND_UNIVERSE + SECTOR_ETFS + MEANREV_UNIVERSE)))
    sections = [
        f"ANOMALY DESK — {datetime.now():%A %b %d, %Y}",
        "=" * 46,
        "\n[1] TREND REGIME (Faber 200DMA)\n" + fmt_trend(trend_screen(closes)),
        "\n[2] SECTOR ROTATION (6mo momentum)\n" + fmt_rotation(rotation_screen(closes)),
        "\n[5] PULLBACK REVERSION (RSI-2, above-trend only)\n" + fmt_meanrev(meanrev_screen(closes)),
    ]
    if os.environ.get("ANTHROPIC_API_KEY"):
        for name, prompt in [("[3] QUALITY COMPOUNDERS", quality_prompt()),
                             ("[4] EARNINGS DRIFT (PEAD)", PEAD_PROMPT)]:
            try:
                sections.append(f"\n{name}\n" + claude_research(prompt))
            except Exception as e:                     # research legs are best-effort
                sections.append(f"\n{name}\n  research call failed: {e}")
                if "credit balance is too low" in str(e).lower():
                    warnings.append(f"{name}: Anthropic credit balance exhausted"
                                    " — top up at console.anthropic.com (Plans & Billing)")
                else:
                    warnings.append(f"{name}: research call failed ({type(e).__name__})")
    else:
        sections.append("\n[3][4] quality + PEAD skipped (set ANTHROPIC_API_KEY)")
    sections.append("\n--\nResearch screen, not advice. Verify before acting.")
    if warnings:                                       # surface degraded runs up top
        banner = "\n".join([
            "!" * 46,
            "⚠  DEGRADED RUN — some research legs did not complete:",
            *(f"     • {w}" for w in warnings),
            "   (price-math strategies [1]/[2]/[5] below are unaffected)",
            "!" * 46,
        ])
        sections.insert(2, "\n" + banner)              # after title + divider
    return "\n".join(sections), warnings


def maybe_email(report: str, warnings: list[str]) -> None:
    to, frm, pw = (os.environ.get(k) for k in
                   ("DESK_EMAIL_TO", "DESK_EMAIL_FROM", "DESK_EMAIL_PASS"))
    if not (to and frm and pw):
        return
    msg = MIMEText(f"<pre style='font-family:monospace'>{report}</pre>", "html")
    flag = "⚠ " if warnings else ""                    # visible in inbox subject line
    msg["Subject"] = f"{flag}Anomaly Desk — {datetime.now():%b %d}"
    msg["From"], msg["To"] = frm, to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(frm, pw)
        s.send_message(msg)


if __name__ == "__main__":
    rep, warnings = build_report()
    print(rep)
    try:
        maybe_email(rep, warnings)
    except Exception as e:
        print(f"\n(email failed: {e})", file=sys.stderr)
