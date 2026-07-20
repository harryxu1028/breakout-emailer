"""
Breakout Emailer
----------------
Scans sector watchlists for stocks that made a FRESH 3-year high yesterday
(yesterday's close broke above the trailing 3y max; the day before did not),
then emails a report with each name grouped by sector and a 5-year price
chart under each ticker (or full history since IPO if shorter).

Sends nothing if there are no breakouts.

Env vars required (set as GitHub Actions secrets or locally):
  EMAIL_FROM          sender address (e.g. yourname@gmail.com)
  EMAIL_TO            recipient address
  GMAIL_APP_PASSWORD  Gmail app password (Google Account > Security > App passwords)

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECTORS = {
    "Software": ["CRM", "NOW", "HUBS", "MNDY", "PCOR"],
    "Payments/Fintech": ["FI", "FOUR", "AFRM", "TOST", "FLYW"],
    "Semis/Memory": ["SNDK", "MU", "WDC", "MRVL", "STX"],
    "Internet": ["SE", "CPNG", "RBLX", "DUOL", "SHOP"],
}

LOOKBACK = 756           # ~3 years of trading days for the high test
MIN_HISTORY = 600        # min trading days to qualify for the breakout test
CHART_YEARS = 5          # chart window (falls back to full history if shorter)
USE_INTRADAY_HIGH = False  # True = test against 3y max of daily HIGHS
SEND_IF_EMPTY = False    # True = send a "no breakouts" email anyway

SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465


# ---------------------------------------------------------------------------
# Company names (looked up only for hits, so at most a handful of requests)
# ---------------------------------------------------------------------------
def get_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
def find_breakouts(px: pd.DataFrame, tickers: list[str]) -> list[dict]:
    closes = px["Close"]
    ref = px["High"] if USE_INTRADAY_HIGH else px["Close"]
    if isinstance(closes, pd.Series):
        closes, ref = closes.to_frame(tickers[0]), ref.to_frame(tickers[0])

    hits = []
    for t in tickers:
        s_close = closes[t].dropna()
        s_ref = ref[t].dropna()
        if len(s_close) < MIN_HISTORY + 2:
            continue

        y_close = s_close.iloc[-1]                        # yesterday
        prior_max_y = s_ref.iloc[-1 - LOOKBACK:-1].max()  # 3y max excl. yesterday
        d2_close = s_close.iloc[-2]
        prior_max_d2 = s_ref.iloc[-2 - LOOKBACK:-2].max()

        if y_close > prior_max_y and d2_close <= prior_max_d2:
            sector = next(sec for sec, lst in SECTORS.items() if t in lst)
            hits.append({
                "ticker": t,
                "name": get_company_name(t),
                "sector": sector,
                "close": y_close,
                "prior_3y_high": prior_max_y,
                "breakout_pct": (y_close / prior_max_y - 1) * 100,
                "date": s_close.index[-1].date().isoformat(),
                "series": s_close,  # reused for charting
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
# Email assembly
# ---------------------------------------------------------------------------
def build_email(hits: list[dict]) -> MIMEMultipart:
    date_str = hits[0]["date"] if hits else pd.Timestamp.today().date().isoformat()
    msg = MIMEMultipart("related")
    msg["Subject"] = (
        f"Breakout Scan {date_str}: {len(hits)} fresh 3y high(s)"
        if hits else f"Breakout Scan {date_str}: no breakouts"
    )
    msg["From"] = os.environ["EMAIL_FROM"]
    msg["To"] = os.environ["EMAIL_TO"]

    parts = [
        "<html><body style='font-family:Arial,Helvetica,sans-serif;color:#222;'>",
        f"<h2 style='margin-bottom:2px;'>Fresh 3-Year-High Breakouts</h2>",
        f"<p style='color:#666;margin-top:0;'>As of close {date_str}</p>",
    ]

    if not hits:
        parts.append("<p>No fresh breakouts today.</p>")
    else:
        by_sector: dict[str, list[dict]] = {}
        for h in hits:
            by_sector.setdefault(h["sector"], []).append(h)

        img_idx = 0
        images = []
        for sector in SECTORS:  # preserve config order
            if sector not in by_sector:
                continue
            parts.append(
                f"<h3 style='border-bottom:1px solid #ddd;"
                f"padding-bottom:4px;'>{sector}</h3>"
            )
            for h in sorted(by_sector[sector],
                            key=lambda x: -x["breakout_pct"]):
                cid = f"chart{img_idx}"
                img_idx += 1
                parts.append(
                    f"<p style='margin:14px 0 2px;font-size:16px;'>"
                    f"<b>{h['ticker']}</b> &nbsp;&middot;&nbsp; "
                    f"{h.get('name', h['ticker'])}</p>"
                    f"<p style='margin:0 0 4px;color:#444;'>"
                    f"${h['close']:,.2f} &nbsp;|&nbsp; "
                    f"prior 3y high ${h['prior_3y_high']:,.2f} &nbsp;|&nbsp; "
                    f"broke out by {h['breakout_pct']:+.2f}%</p>"
                    f"<img src='cid:{cid}' width='620' "
                    f"style='display:block;margin-bottom:14px;'>"
                )
                images.append((cid, make_chart_png(h)))

    parts.append("</body></html>")
    msg.attach(MIMEText("".join(parts), "html"))

    if hits:
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
    tickers = sorted({t for lst in SECTORS.values() for t in lst})
    px = yf.download(tickers, period="6y", auto_adjust=True, progress=False)
    hits = find_breakouts(px, tickers)

    if not hits and not SEND_IF_EMPTY:
        print("No breakouts — no email sent.")
        return

    send(build_email(hits))
    print(f"Email sent: {len(hits)} breakout(s) — "
          + ", ".join(h["ticker"] for h in hits))


if __name__ == "__main__":
    sys.exit(main())
