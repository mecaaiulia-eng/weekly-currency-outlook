import argparse
import base64
import html
import io
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

DOCS = Path("docs")
ARCHIVE = DOCS / "archive"

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
    "comm_positions_long_all",
    "comm_positions_short_all",
    "noncomm_positions_long_all",
    "noncomm_positions_short_all",
]


def demo_rows():
    """Sample COT values for previewing the page layout."""
    return [
        {
            "market_and_exchange_names": name,
            "report_date_as_yyyy_mm_dd": "2026-11-17T00:00:00.000",
            "comm_positions_long_all": "120000",
            "comm_positions_short_all": "95000",
            "noncomm_positions_long_all": "85000",
            "noncomm_positions_short_all": "110000",
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


def demo_prices(days=15 * 366):
    """Deterministic synthetic prices for previewing charts only."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=15)
    dates = pd.date_range(start=start, end=end, freq="B")
    i = np.arange(len(dates))

    # Synthetic drift plus a repeating seasonal-shaped component.
    returns = (
        0.00008
        + 0.0015 * np.sin(2 * np.pi * i / 252)
        + 0.0005 * np.sin(2 * np.pi * i / 37)
    )
    values = 1.0 * np.exp(np.cumsum(returns))
    return pd.Series(values, index=dates, name="Close")


def get_weekly_prices(ticker, demo=False):
    """Weekly closes for the report's one-year return and monthly table."""
    if demo:
        prices = demo_prices()
        return prices.resample("W-FRI").last().dropna()

    import yfinance as yf

    data = yf.download(
        ticker,
        period="5y",
        interval="1wk",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        return pd.Series(dtype=float)

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def get_daily_prices(ticker, demo=False):
    """Daily closes used to calculate the price-seasonality chart."""
    if demo:
        return demo_prices()

    import yfinance as yf

    data = yf.download(
        ticker,
        period="15y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        return pd.Series(dtype=float)

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def seasonality_chart(prices, demo=False):
    """
    Compare average historical year-to-date price returns with the current year.

    Each year's first available close is the baseline (0%). Leap days are
    adjusted so dates after February line up consistently across years.
    """
    if prices.empty:
        return "<p class='muted'>No daily price data for the seasonality chart.</p>"

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    current_year = pd.Timestamp.today().year
    frame = pd.DataFrame({"close": prices})
    frame["year"] = frame.index.year

    # Align leap years with non-leap years after February 28.
    day = frame.index.dayofyear.to_numpy()
    leap_adjustment = (
        frame.index.is_leap_year
        & (frame.index.month.to_numpy() > 2)
    )
    frame["season_day"] = day - leap_adjustment.astype(int)

    # Normalize each year's prices to its first available close.
    frame["year_start"] = frame.groupby("year")["close"].transform("first")
    frame["return_pct"] = (frame["close"] / frame["year_start"] - 1) * 100

    history = frame[
        (frame["year"] >= current_year - 15)
        & (frame["year"] < current_year)
    ]
    current = frame[frame["year"] == current_year]

    if history.empty or current.empty:
        return (
            "<p class='muted'>Not enough daily price history to compare "
            "this year with prior years.</p>"
        )

    typical = history.groupby("season_day")["return_pct"].mean()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(
        typical.index,
        typical.values,
        color="#929292",
        linewidth=2.5,
        label="Typical (15-year average)",
    )
    ax.plot(
        current["season_day"],
        current["return_pct"],
        color="#222222",
        linewidth=2,
        label=f"{current_year} so far",
    )

    month_starts = pd.date_range(
        f"{current_year}-01-01",
        f"{current_year}-12-01",
        freq="MS",
    )
    tick_days = []
    tick_labels = []
    for date in month_starts:
        adjusted_day = date.dayofyear
        if date.is_leap_year and date.month > 2:
            adjusted_day -= 1
        tick_days.append(adjusted_day)
        tick_labels.append(date.strftime("%b"))

    ax.set_xticks(tick_days)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Cumulative price return (%)")
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.6)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    data_label = "Synthetic demo data" if demo else "Historical price data"

    return (
        "<img class='season-chart' "
        "alt='Average historical seasonal price returns compared with this year' "
        f"src='data:image/png;base64,{image_data}'>"
        f"<p class='muted'>{data_label}. Each year is rebased to its first "
        "available close. Historical patterns are descriptive, not forecasts.</p>"
    )


def numeric(row, field):
    value = pd.to_numeric(row.get(field), errors="coerce")
    return float(value) if pd.notna(value) else None


def market_row(cot, market_name):
    matched = cot[
        cot["market_and_exchange_names"]
        .fillna("")
        .str.contains(market_name, case=False, regex=False)
    ].copy()

    if matched.empty:
        return None

    matched["report_date_as_yyyy_mm_dd"] = pd.to_datetime(
        matched["report_date_as_yyyy_mm_dd"],
        errors="coerce",
    )
    matched = matched.dropna(subset=["report_date_as_yyyy_mm_dd"])
    if matched.empty:
        return None

    row = matched.sort_values("report_date_as_yyyy_mm_dd").iloc[-1]

    comm_long = numeric(row, "comm_positions_long_all")
    comm_short = numeric(row, "comm_positions_short_all")
    spec_long = numeric(row, "noncomm_positions_long_all")
    spec_short = numeric(row, "noncomm_positions_short_all")

    return {
        "date": row["report_date_as_yyyy_mm_dd"].date().isoformat(),
        "commercial_net": (
            comm_long - comm_short
            if comm_long is not None and comm_short is not None
            else None
        ),
        "speculator_net": (
            spec_long - spec_short
            if spec_long is not None and spec_short is not None
            else None
        ),
    }


def monthly_seasonality(prices):
    """Descriptive historical average monthly return summary."""
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
        result.append(
            {
                "month": int(month),
                "avg": float(group["return"].mean()),
                "win_rate": float((group["return"] > 0).mean()),
                "n": int(group["return"].count()),
            }
        )

    return sorted(result, key=lambda item: item["month"])


def commentary(name, cot_row, latest_price, one_year_return):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return (
            "Automated summary: use the positioning and price figures below "
            "as descriptive context. This report does not establish a "
            "trading signal."
        )

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        prompt = (
            f"Write 2 concise, neutral sentences for a market research report "
            f"about {name}. "
            f"CFTC commercial net: {cot_row.get('commercial_net')}; "
            f"noncommercial net: {cot_row.get('speculator_net')}; "
            f"latest weekly close: {latest_price}; "
            f"approx. one-year return: {one_year_return}. "
            "Do not give trading instructions or make predictions. "
            "Mention data limitations."
        )
        result = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return result.content[0].text
    except Exception as exc:
        return f"Commentary unavailable ({html.escape(str(exc)[:160])})."


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
        weekly_prices = get_weekly_prices(ticker, demo)
        daily_prices = get_daily_prices(ticker, demo)

        latest = (
            float(weekly_prices.iloc[-1])
            if not weekly_prices.empty
            else None
        )
        one_year_return = (
            float(weekly_prices.iloc[-1] / weekly_prices.iloc[-53] - 1)
            if len(weekly_prices) >= 53
            else None
        )

        if demo:
            comment = "Demo report: sample figures and synthetic prices."
        else:
            comment = commentary(
                name,
                positions or {},
                latest,
                one_year_return,
            )

        rows.append(
            {
                "name": name,
                "ticker": ticker,
                "positions": positions,
                "latest": latest,
                "one_year_return": one_year_return,
                "monthly_seasonality": monthly_seasonality(weekly_prices),
                "season_chart": seasonality_chart(daily_prices, demo=demo),
                "comment": comment,
            }
        )

    cards = []

    for item in rows:
        positions = item["positions"] or {}

        monthly_rows = "".join(
            (
                f"<tr><td>{pd.Timestamp(2000, month['month'], 1).strftime('%B')}</td>"
                f"<td>{fmt_pct(month['avg'])}</td>"
                f"<td>{fmt_pct(month['win_rate'])}</td>"
                f"<td>{month['n']}</td></tr>"
            )
            for month in item["monthly_seasonality"]
        ) or "<tr><td colspan='4'>Not enough price data</td></tr>"

        latest_display = (
            "N/A" if item["latest"] is None else f"{item['latest']:.5g}"
        )

        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(item['name'])}</h2>
              <p class="muted">
                {html.escape(item['ticker'])}
                · COT report date: {html.escape(positions.get('date', 'N/A'))}
              </p>

              <div class="stats">
                <div><small>Commercial net</small>
                  <strong>{fmt_number(positions.get('commercial_net'))}</strong>
                </div>
                <div><small>Noncommercial net</small>
                  <strong>{fmt_number(positions.get('speculator_net'))}</strong>
                </div>
                <div><small>Latest weekly close</small>
                  <strong>{latest_display}</strong>
                </div>
                <div><small>Approx. 1-year return</small>
                  <strong>{fmt_pct(item['one_year_return'])}</strong>
                </div>
              </div>

              <h3>Seasonal price movement</h3>
              {item['season_chart']}

              <p>{html.escape(item['comment'])}</p>

              <details>
                <summary>Historical calendar-month returns</summary>
                <table>
                  <thead>
                    <tr>
                      <th>Month</th>
                      <th>Average return</th>
                      <th>Positive months</th>
                      <th>Cases</th>
                    </tr>
                  </thead>
                  <tbody>{monthly_rows}</tbody>
                </table>
                <p class="muted">
                  Calculated from available weekly prices and grouped by
                  calendar month. Small samples can be misleading.
                </p>
              </details>
            </section>
            """
        )

    demo_notice = (
        "<p class='warning'><strong>Demo:</strong> figures and charts use "
        "sample or synthetic data, not live market data.</p>"
        if demo
        else ""
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weekly Currency Outlook</title>
<style>
body {{
  font: 16px system-ui, sans-serif;
  max-width: 1000px;
  margin: 2rem auto;
  padding: 0 1rem;
  color: #202938;
  background: #f5f7fa;
}}
header, .card {{
  background: white;
  border: 1px solid #dce2ea;
  border-radius: 12px;
  padding: 1.2rem;
  margin: 1rem 0;
}}
h1 {{ margin-bottom: .2rem; }}
h3 {{ margin-top: 1.5rem; }}
.muted, small {{ color: #657386; }}
.stats {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: .8rem;
}}
.stats div {{ background: #f5f7fa; padding: .8rem; border-radius: 8px; }}
.stats strong {{ display: block; margin-top: .3rem; }}
.season-chart {{ display: block; width: 100%; height: auto; margin: .5rem 0; }}
table {{ border-collapse: collapse; width: 100%; margin-top: .8rem; }}
td, th {{ padding: .45rem; border-bottom: 1px solid #e3e8ef; text-align: left; }}
.warning {{ font-size: .9rem; color: #536174; }}
</style>
</head>
<body>
<header>
  <h1>Weekly Currency Outlook</h1>
  <p>Generated {now.strftime('%Y-%m-%d %H:%M UTC')}</p>
  {demo_notice}
  <p class="warning">
    Research information only—not financial advice. COT reflects reported
    positions, not trader intent. Historical seasonal price movement is
    descriptive, not a forecast. Price data is supplied by Yahoo Finance
    and may be incomplete.
  </p>
</header>
{''.join(cards)}
</body>
</html>
"""

    (DOCS / "index.html").write_text(page, encoding="utf-8")
    archive_name = f"{now.date().isoformat()}.html"
    (ARCHIVE / archive_name).write_text(page, encoding="utf-8")
    print("Wrote docs/index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build with sample values and synthetic prices.",
    )
    args = parser.parse_args()
    build(demo=args.demo)
