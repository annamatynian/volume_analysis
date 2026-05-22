"""
download_oi_metrics.py
======================
Download daily BTCUSDT metrics from Binance Vision.
Date range: 2025-01-01 to 2026-04-01 (~457 files, ~5MB total).

Saves daily close-of-day OI to: data/metrics/BTCUSDT-metrics-daily.csv

Run: python download_oi_metrics.py
"""

import urllib.request
import zipfile
import io
import csv
import os
import time
from datetime import date, timedelta

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/"
OUT_DIR  = "data/metrics"
OUT_FILE = os.path.join(OUT_DIR, "BTCUSDT-metrics-daily.csv")

START_DATE = date(2025, 1, 1)
END_DATE   = date(2026, 4, 1)

os.makedirs(OUT_DIR, exist_ok=True)

COLUMNS = [
    'date',
    'sum_open_interest',
    'sum_open_interest_value',
    'count_toptrader_long_short_ratio',
    'sum_toptrader_long_short_ratio',
    'count_long_short_ratio',
    'sum_taker_long_short_vol_ratio',
]

# Build list of dates
all_dates = []
d = START_DATE
while d <= END_DATE:
    all_dates.append(d)
    d += timedelta(days=1)

print(f"Dates to download: {len(all_dates)}")
print(f"From {START_DATE} to {END_DATE}")
print(f"Output: {OUT_FILE}")
print()

rows_out = []
errors   = []
skipped  = 0

for i, day in enumerate(all_dates):
    date_str = day.strftime("%Y-%m-%d")
    filename = f"BTCUSDT-metrics-{date_str}.zip"
    url      = BASE_URL + filename

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read()

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                lines = f.read().decode('utf-8').splitlines()

        reader = list(csv.reader(lines))
        if len(reader) < 2:
            skipped += 1
            continue

        header   = reader[0]
        last_row = reader[-1]   # last 5-min interval = end-of-day OI

        def get(col):
            if col in header:
                return last_row[header.index(col)]
            return ''

        rows_out.append({
            'date':                            date_str,
            'sum_open_interest':               get('sum_open_interest'),
            'sum_open_interest_value':         get('sum_open_interest_value'),
            'count_toptrader_long_short_ratio': get('count_toptrader_long_short_ratio'),
            'sum_toptrader_long_short_ratio':  get('sum_toptrader_long_short_ratio'),
            'count_long_short_ratio':          get('count_long_short_ratio'),
            'sum_taker_long_short_vol_ratio':  get('sum_taker_long_short_vol_ratio'),
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(all_dates)}] {date_str} OK  (total saved: {len(rows_out)})")

        time.sleep(0.1)   # polite rate limit

    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Weekend or missing day — normal for futures metrics
            skipped += 1
        else:
            errors.append((date_str, str(e)))
            print(f"  [{i+1:3d}] {date_str} HTTP ERROR {e.code}")
    except Exception as e:
        errors.append((date_str, str(e)))
        print(f"  [{i+1:3d}] {date_str} ERROR: {e}")

# Write output CSV
with open(OUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows_out)

print()
print("=" * 60)
print(f"DONE")
print(f"  Saved rows : {len(rows_out)}")
print(f"  Skipped    : {skipped}  (404 = weekend/missing)")
print(f"  Errors     : {len(errors)}")
print(f"  Output     : {OUT_FILE}")
if errors:
    print("  Error list:")
    for d, e in errors[:10]:
        print(f"    {d}: {e}")
