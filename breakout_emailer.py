"""
Breakout Emailer — sector-wide edition
--------------------------------------
Pulls ALL US-listed equities in the configured Yahoo Finance sectors
(via yfinance's screener API), scans for stocks that made a FRESH
3-year high yesterday, and emails a report: names grouped by sector,
each with ticker, company name, stats, and a 5-year price chart
inline in the email body (full history since IPO if shorter).

Sends nothing if there are no breakouts.

Env vars required:
  EMAIL_FROM          sender address (e.g. yourname@gmail.com)
  EMAIL_TO            recipient address
  GMAIL_APP_PASSWORD  Gmail app password

Usage:
  pip install yfinance pandas matplotlib
  python breakout_emailer.py
"""

import io
import os
import smtplib
import sys
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Yahoo Finance sector names (exact strings). Valid options:
#   "Technology", "Financial Services", "Communication Services",
#   "Consumer Cyclical", "Consumer Defensive", "Healthcare",
#   "Industrials", "Energy", "Basic Materials", "Real Estate", "Utilities"
SECTORS = [
    "Technology",
    "Financial Services",
    "Communication Services",
    "Consumer Cyclical",
]

REGION = "us"                    # listing region filter
MIN_MARKET_CAP = 300_000_000     # $300M floor; set to 0 for truly everything
MAX_NAMES_PER_SECTOR = 2000      # hard safety cap per sector

# Sector ETF dashboard: a 1-year chart for each is appended to every
# email. Tickers with no Yahoo data are silently skipped.
ETF_TICKERS = ["SOXX", "IGV", "IHAK", "IDGT", "IYF", "KRE", "IAI",
               "IYJ", "ITA", "IYT", "IFRA", "IYC", "ITB", "IYH", "POWR"]

LOOKBACK = 756                   # ~3 years of trading days for the high test
MIN_HISTORY = 600                # min trading days to qualify
CHART_YEARS = 5                  # chart window (full history if shorter)
USE_INTRADAY_HIGH = False        # True = test against 3y max of daily HIGHS
SEND_IF_EMPTY = False            # True = send a "no breakouts" email anyway

MAX_CHARTS = 40                  # cap inline charts per email (Gmail clips
                                 # huge messages); overflow listed as text
DOWNLOAD_CHUNK = 200             # tickers per yf.download batch

SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
PAGE = 250                       # Yahoo screener max page size


# ---------------------------------------------------------------------------
# Universe: enumerate every ticker Yahoo classifies in each sector
# ---------------------------------------------------------------------------
def build_universe() -> dict[str, dict]:
    """Returns {ticker: {"sector": ..., "name": ...}} across all SECTORS."""
    universe: dict[str, dict] = {}
    for sector in SECTORS:
        clauses = [
            EquityQuery("eq", ["sector", sector]),
            EquityQuery("eq", ["region", REGION]),
        ]
        if MIN_MARKET_CAP > 0:
            clauses.append(
                EquityQuery("gte", ["intradaymarketcap", MIN_MARKET_CAP])
            )
        query = EquityQuery("and", clauses)

        offset, total, fetched = 0, None, 0
        while True:
            try:
                res = yf.screen(
                    query, size=PAGE, offset=offset,
                    sortField="intradaymarketcap", sortAsc=False,
                )
            except Exception as e:
                print(f"[warn] screener page failed "
                      f"({sector} offset {offset}): {e}")
                break

            quotes = res.get("quotes", []) if res else []
            if total is None:
                total = res.get("total", 0)
                print(f"{sector}: {total} names pass filters")
            if not quotes:
                break

            for q in quotes:
                sym = q.get("symbol")
                if not sym or "." in sym or "^" in sym:
                    continue  # skip odd share classes / indices
                if q.get("quoteType") != "EQUITY":
                    continue  # single stocks only — no mutual funds/ETFs
                universe.setdefault(sym, {
                    "sector": sector,
                    "name": q.get("shortName")
                            or q.get("longName") or sym,
                })
            fetched += len(quotes)
            offset += PAGE
            if fetched >= min(total, MAX_NAMES_PER_SECTOR):
                break
    print(f"Universe: {len(universe)} unique tickers "
          f"across {len(SECTORS)} sectors")
    return universe


# ---------------------------------------------------------------------------
# Prices: chunked batch download
# ---------------------------------------------------------------------------
def download_prices(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes, highs = [], []
    for i in range(0, len(tickers), DOWNLOAD_CHUNK):
        chunk = tickers[i:i + DOWNLOAD_CHUNK]
        px = yf.download(chunk, period="6y", auto_adjust=True,
                         progress=False, threads=True)
        c, h = px["Close"], px["High"]
        if isinstance(c, pd.Series):  # single-ticker chunk edge case
            c, h = c.to_frame(chunk[0]), h.to_frame(chunk[0])
        closes.append(c)
        highs.append(h)
        print(f"  downloaded {min(i + DOWNLOAD_CHUNK, len(tickers))}"
              f"/{len(tickers)}")
    return (pd.concat(closes, axis=1),
            pd.concat(highs, axis=1))


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
def find_breakouts(closes: pd.DataFrame, highs: pd.DataFrame,
                   universe: dict[str, dict]) -> list[dict]:
    ref_df = highs if USE_INTRADAY_HIGH else closes

    # A row dated today is only a COMPLETED close if the market has
    # closed (>= ~4:05pm ET). Before that, drop it and scan the prior
    # trading day; after the close, keep it so a 4:30pm run scans
    # today's close.
    now_et = pd.Timestamp.now(tz="America/New_York")
    market_closed = (now_et.hour, now_et.minute) >= (16, 5)
    if (len(closes) and closes.index[-1].date() >= now_et.date()
            and not market_closed):
        closes, ref_df = closes.iloc[:-1], ref_df.iloc[:-1]

    hits = []
    for t, meta in universe.items():
        if t not in closes.columns:
            continue
        s_close = closes[t].dropna()
        s_ref = ref_df[t].dropna()
        if len(s_close) < MIN_HISTORY + 2:
            continue

        y_close = s_close.iloc[-1]                        # yesterday
        prior_max_y = s_ref.iloc[-1 - LOOKBACK:-1].max()  # 3y max excl. yday
        d2_close = s_close.iloc[-2]
        prior_max_d2 = s_ref.iloc[-2 - LOOKBACK:-2].max()

        if y_close > prior_max_y and d2_close <= prior_max_d2:
            hits.append({
                "ticker": t,
                "name": meta["name"],
                "sector": meta["sector"],
                "close": y_close,
                "prior_3y_high": prior_max_y,
                "breakout_pct": (y_close / prior_max_y - 1) * 100,
                "date": s_close.index[-1].date().isoformat(),
                "series": s_close,
            })
    return hits


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def make_chart_png(hit: dict) -> bytes:
    s = hit["series"]
    cutoff = s.index[-1] - pd.DateOffset(years=CHART_YEARS)
    s5 = s[s.index >= cutoff]  # full history if < 5y available

    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=110)
    ax.plot(s5.index, s5.values, linewidth=1.4, color="#1a4d8f")
    ax.axhline(hit["prior_3y_high"], linestyle="--", linewidth=1,
               color="#c0392b", alpha=0.8, label="prior 3y high")
    ax.plot(s5.index[-1], s5.values[-1], "o", markersize=5, color="#c0392b")

    span_yrs = (s5.index[-1] - s5.index[0]).days / 365.25
    ax.set_title(
        f"{hit['ticker']}  —  {span_yrs:.1f}y history"
        + ("" if span_yrs >= CHART_YEARS - 0.1 else "  (since IPO)"),
        fontsize=10, loc="left",
    )
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ETF dashboard: fixed list, 1-year charts
# ---------------------------------------------------------------------------
def get_etf_sections() -> list[dict]:
    """1y data for the ETF dashboard; skips tickers Yahoo lacks."""
    if not ETF_TICKERS:
        return []
    try:
        px = yf.download(ETF_TICKERS, period="1y", auto_adjust=True,
                         progress=False)
    except Exception as e:
        print(f"[warn] ETF download failed: {e}")
        return []
    closes = px["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(ETF_TICKERS[0])

    out = []
    for t in ETF_TICKERS:  # preserve config order
        if t not in closes.columns:
            continue
        s = closes[t].dropna()
        if len(s) < 30:
            continue  # not on Yahoo / insufficient history
        try:
            info = yf.Ticker(t).info
            name = info.get("shortName") or info.get("longName") or t
        except Exception:
            name = t
        out.append({
            "ticker": t, "name": name, "series": s,
            "ret_1y": (s.iloc[-1] / s.iloc[0] - 1) * 100,
        })
    return out


def make_etf_chart_png(etf: dict) -> bytes:
    s = etf["series"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=110)
    color = "#1a7a3a" if s.iloc[-1] >= s.iloc[0] else "#c0392b"
    ax.plot(s.index, s.values, linewidth=1.4, color=color)
    ax.set_title(f"{etf['ticker']}  —  1y return {etf['ret_1y']:+.1f}%",
                 fontsize=10, loc="left")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Email assembly
# ---------------------------------------------------------------------------
def build_email(hits: list[dict], etfs: list[dict]) -> MIMEMultipart:
    date_str = hits[0]["date"] if hits else \
        pd.Timestamp.today().date().isoformat()
    msg = MIMEMultipart("related")
    msg["Subject"] = (
        f"Breakout Scan {date_str}: {len(hits)} fresh 3y high(s)"
        if hits else f"Breakout Scan {date_str}: no breakouts"
    )
    msg["From"] = os.environ["EMAIL_FROM"]
    msg["To"] = os.environ["EMAIL_TO"]

    parts = [
        "<html><body style='font-family:Arial,Helvetica,sans-serif;"
        "color:#222;'>",
        "<h2 style='margin-bottom:2px;'>Fresh 3-Year-High Breakouts</h2>",
        f"<p style='color:#666;margin-top:0;'>As of close {date_str}</p>",
    ]
    images = []

    if not hits:
        parts.append("<p>No fresh breakouts today.</p>")
    else:
        by_sector: dict[str, list[dict]] = {}
        for h in hits:
            by_sector.setdefault(h["sector"], []).append(h)

        img_idx = 0
        overflow: list[dict] = []
        for sector in SECTORS:  # preserve config order
            if sector not in by_sector:
                continue
            parts.append(
                f"<h3 style='border-bottom:1px solid #ddd;"
                f"padding-bottom:4px;'>{sector}</h3>"
            )
            for h in sorted(by_sector[sector],
                            key=lambda x: -x["breakout_pct"]):
                if img_idx >= MAX_CHARTS:
                    overflow.append(h)
                    continue
                cid = f"chart{img_idx}"
                img_idx += 1
                parts.append(
                    f"<p style='margin:14px 0 2px;font-size:16px;'>"
                    f"<b>{h['ticker']}</b> &nbsp;&middot;&nbsp; "
                    f"{h['name']}</p>"
                    f"<p style='margin:0 0 4px;color:#444;'>"
                    f"${h['close']:,.2f} &nbsp;|&nbsp; "
                    f"prior 3y high ${h['prior_3y_high']:,.2f} &nbsp;|&nbsp; "
                    f"broke out by {h['breakout_pct']:+.2f}%</p>"
                    f"<img src='cid:{cid}' width='620' "
                    f"style='display:block;margin-bottom:14px;'>"
                )
                images.append((cid, make_chart_png(h)))

        if overflow:
            rows = "".join(
                f"<li>{h['ticker']} &middot; {h['name']} "
                f"({h['sector']}, {h['breakout_pct']:+.2f}%)</li>"
                for h in overflow
            )
            parts.append(
                f"<h3>Also broke out (chart cap of {MAX_CHARTS} "
                f"reached)</h3><ul>{rows}</ul>"
            )

    if etfs:
        parts.append(
            "<h2 style='margin-top:28px;border-top:2px solid #ddd;"
            "padding-top:14px;'>Sector ETFs &mdash; trailing 1 year</h2>"
        )
        for i, e in enumerate(etfs):
            cid = f"etf{i}"
            parts.append(
                f"<p style='margin:14px 0 2px;font-size:16px;'>"
                f"<b>{e['ticker']}</b> &nbsp;&middot;&nbsp; {e['name']}</p>"
                f"<img src='cid:{cid}' width='620' "
                f"style='display:block;margin-bottom:14px;'>"
            )
            images.append((cid, make_etf_chart_png(e)))

    parts.append("</body></html>")
    msg.attach(MIMEText("".join(parts), "html"))

    for cid, png in images:
        img = MIMEImage(png, _subtype="png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline",
                       filename=f"{cid}.png")
        msg.attach(img)
    return msg


def send(msg: MIMEMultipart) -> None:
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(os.environ["EMAIL_FROM"],
                     os.environ["GMAIL_APP_PASSWORD"])
        server.send_message(msg)


# ---------------------------------------------------------------------------
def main() -> None:
    universe = build_universe()
    if not universe:
        print("Universe came back empty — screener may be rate-limited. "
              "Aborting without email.")
        return

    closes, highs = download_prices(sorted(universe))
    hits = find_breakouts(closes, highs, universe)

    if not hits and not SEND_IF_EMPTY:
        print("No breakouts — no email sent.")
        return

    etfs = get_etf_sections()
    print(f"ETF dashboard: {len(etfs)}/{len(ETF_TICKERS)} tickers found")
    send(build_email(hits, etfs))
    print(f"Email sent: {len(hits)} breakout(s) — "
          + ", ".join(h["ticker"] for h in hits))


if __name__ == "__main__":
    sys.exit(main())
