import argparse
import html
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DOCS = Path("docs")
ARCHIVE = DOCS / "archive"

# Yahoo Finance currency futures symbols; change this list to suit your coverage.
MARKETS = {
    "Euro FX": "6E=F",
    "British Pound": "6B=F",
    "Japanese Yen": "6J=F",
    "Swiss Franc": "6S=F",
    "Canadian Dollar": "6C=F",
    "Australian Dollar": "6A=F",
    "New Zealand Dollar": "6N=F",
}

CFTC_URL = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
CFTC_FIELDS = [
    "report_date_as_yyyy_mm_dd",
    "market_and_exchange_names",
    "commercial_positions_long_all",
    "commercial_positions_short_all",
    "noncommercial_positions_long_all",
    "noncommercial_positions_short_all",
]

def demo_rows():
    """Offline sample data so the page can be previewed without network access."""
    return [
        {
            "market_and_exchange_names": name,
            "report_date_as_yyyy_mm_dd": "2026-11-17T00:00:00.000",
            "commercial_positions_long_all": "120000",
            "commercial_positions_short_all": "95000",
            "noncommercial_positions_long_all": "85000",
            "noncommercial_positions_short_all": "110000",
        }
        for name in MARKETS
    ]

def get_cot(demo=False):
    if demo:
        return pd.DataFrame(demo_rows())

    response = requests.get(
        CFTC_URL,
        params={
            "$select": ",".join(CFTC_FIELDS),
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": 50000,
        },
        timeout=45,
        headers={"User-Agent": "weekly-currency-outlook/1.0"},
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise RuntimeError("CFTC returned no rows.")
    return pd.DataFrame(rows)

def get_prices(ticker, demo=False):
    if demo:
        return pd.Series(
            [1.00 + i * 0.002 + ((i % 5) - 2) * 0.001 for i in range(260)],
            index=pd.date_range(end=pd.Timestamp.today(), periods=260, freq="W"),
            name="Close",
        )

    import yfinance as yf

    data = yf.download(
        ticker, period="5y", interval="1wk", auto_adjust=True,
        progress=False, threads=False
    )
    if data.empty:
        return pd.Series(dtype=float)

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()

def numeric(row, field):
    value = pd.to_numeric(row.get(field), errors="coerce")
    return float(value) if pd.notna(value) else None

def market_row(cot, market_name):
    matched = cot[
        cot["market_and_exchange_names"].fillna("").str.contains(
            market_name, case=False, regex=False
        )
    ].copy()

    if matched.empty:
        return None

    matched["report_date_as_yyyy_mm_dd"] = pd.to_datetime(
        matched["report_date_as_yyyy_mm_dd"], errors="coerce"
    )
    matched = matched.dropna(subset=["report_date_as_yyyy_mm_dd"])
    if matched.empty:
        return None

    row = matched.sort_values("report_date_as_yyyy_mm_dd").iloc[-1]
    comm_long = numeric(row, "commercial_positions_long_all")
    comm_short = numeric(row, "commercial_positions_short_all")
    spec_long = numeric(row, "noncommercial_positions_long_all")
    spec_short = numeric(row, "noncommercial_positions_short_all")

    return {
        "date": row["report_date_as_yyyy_mm_dd"].date().isoformat(),
        "commercial_net": (
            comm_long - comm_short
            if comm_long is not None and comm_short is not None else None
        ),
        "speculator_net": (
            spec_long - spec_short
            if spec_long is not None and spec_short is not None else None
        ),
    }

def seasonality(prices):
    """Simple historical calendar-month return summary; descriptive, not predictive."""
    if prices.empty:
        return []

    monthly = prices.resample("ME").last().dropna()
    returns = monthly.pct_change().dropna()
    if returns.empty:
        return []

    frame = pd.DataFrame({"return": returns})
    frame["month"] = frame.index.month
    result = []
    for month, group in frame.groupby("month"):
        result.append({
            "month": int(month),
            "avg": float(group["return"].mean()),
            "win_rate": float((group["return"] > 0).mean()),
            "n": int(group["return"].count()),
        })
    return sorted(result, key=lambda x: x["month"])

def commentary(name, cot_row, latest_price, one_year_return):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return (
            "Automated summary: use the positioning and price figures below as "
            "descriptive context. This report does not establish a trading signal."
        )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        prompt = (
            f"Write 2 concise, neutral sentences for a market research report about {name}. "
            f"CFTC commercial net: {cot_row.get('commercial_net')}; "
            f"noncommercial net: {cot_row.get('speculator_net')}; "
            f"latest weekly close: {latest_price}; "
            f"approx. one-year return: {one_year_return}. "
            "Do not give trading instructions or make predictions. Mention data limitations."
        )
        result = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return result.content[0].text
    except Exception as exc:
        return f"Commentary unavailable ({html.escape(str(exc))[:160]})."

def fmt_number(value):
    return "N/A" if value is None or pd.isna(value) else f"{value:,.0f}"

def fmt_pct(value):
    return "N/A" if value is None or pd.isna(value) else f"{value:.1%}"

def build(demo=False):
    DOCS.mkdir(exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    cot = get_cot(demo)
    now = datetime.now(timezone.utc)
    rows = []

    for name, ticker in MARKETS.items():
        positions = market_row(cot, name)
        prices = get_prices(ticker, demo)
        latest = float(prices.iloc[-1]) if not prices.empty else None
        yr_return = (
            float(prices.iloc[-1] / prices.iloc[-53] - 1)
            if len(prices) >= 53 else None
        )
        seasonal = seasonality(prices)

        rows.append({
            "name": name,
            "ticker": ticker,
            "positions": positions,
            "latest": latest,
            "yr_return": yr_return,
            "seasonality": seasonal,
            "comment": commentary(name, positions or {}, latest, yr_return),
        })

    cards = []
    for item in rows:
        pos = item["positions"] or {}
        seasons = "".join(
            f"<tr><td>{pd.Timestamp(2000, s['month'], 1).strftime('%B')}</td>"
            f"<td>{fmt_pct(s['avg'])}</td><td>{fmt_pct(s['win_rate'])}</td>"
            f"<td>{s['n']}</td></tr>"
            for s in item["seasonality"]
        ) or "<tr><td colspan='4'>Not enough price data</td></tr>"

        cards.append(f"""
        <section class="card">
          <h2>{html.escape(item['name'])}</h2>
          <p class="muted">{html.escape(item['ticker'])} · COT report date: {html.escape(pos.get('date', 'N/A'))}</p>
          <div class="stats">
            <div><small>Commercial net</small><strong>{fmt_number(pos.get('commercial_net'))}</strong></div>
            <div><small>Noncommercial net</small><strong>{fmt_number(pos.get('speculator_net'))}</strong></div>
            <div><small>Latest weekly close</small><strong>{'N/A' if item['latest'] is None else f"{item['latest']:.5g}"}</strong></div>
            <div><small>Approx. 1-year return</small><strong>{fmt_pct(item['yr_return'])}</strong></div>
          </div>
          <p>{html.escape(item['comment'])}</p>
          <details><summary>Historical calendar-month returns</summary>
            <table><thead><tr><th>Month</th><th>Average</th><th>Positive years</th><th>Cases</th></tr></thead>
            <tbody>{seasons}</tbody></table>
            <p class="muted">Calculated from available weekly prices, aggregated by calendar month. Small samples can be misleading.</p>
          </details>
        </section>""")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weekly Currency Outlook</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#202938;background:#f5f7fa}}
header,.card{{background:white;border:1px solid #dce2ea;border-radius:12px;padding:1.2rem;margin:1rem 0}}
h1{{margin-bottom:.2rem}} .muted,small{{color:#657386}} .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.8rem}}
.stats div{{background:#f5f7fa;padding:.8rem;border-radius:8px}} .stats strong{{display:block;margin-top:.3rem}}
table{{border-collapse:collapse;width:100%;margin-top:.8rem}}td,th{{padding:.45rem;border-bottom:1px solid #e3e8ef;text-align:left}}
.warning{{font-size:.9rem;color:#536174}}
</style></head><body>
<header><h1>Weekly Currency Outlook</h1>
<p>Generated {now.strftime('%Y-%m-%d %H:%M UTC')}</p>
<p class="warning">Research information only—not financial advice. COT reflects reported positions, not trader intent; historical seasonality is not a forecast. Price data is supplied by Yahoo Finance and may be incomplete.</p></header>
{''.join(cards)}
</body></html>"""

    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (ARCHIVE / f"{now.date().isoformat()}.html").write_text(page, encoding="utf-8")
    print("Wrote docs/index.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Build with sample values.")
    args = parser.parse_args()
    build(demo=args.demo)
