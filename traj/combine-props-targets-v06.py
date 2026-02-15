import pandas as pd
import numpy as np
import re

# ---------------- Read target data ----------------
target = pd.read_csv('targets/output.txt', delimiter=' ', header=0, index_col=0)

select_dG = 2
dG = target.iloc[:, select_dG]
dG.index = dG.index.astype(str)

def _hnum(x):
    m = re.match(r'^H(\d+)$', str(x).strip())
    return int(m.group(1)) if m else None

rows = []

for host_name in dG.index:
    try:
        # IMPORTANT: do NOT use index_col=0 here; we want Real_Name + paper_name as columns
        all_prop = pd.read_csv(
            f'all-host-props-nopore/host_{host_name}_all_props.csv',
            delimiter=',',
            header=0
        )
    except FileNotFoundError:
        print(f'Property file not found. Skipping host: {host_name}')
        continue

    if all_prop.shape[0] < 1:
        print(f'Empty property file. Skipping host: {host_name}')
        continue

    # Take the single row as a dict
    row = all_prop.iloc[0].to_dict()

    # --------- Fix/standardize name columns ----------
    # If both exist, prefer 'Real_Name' and drop 'real_name'
    if 'Real_Name' in row and 'real_name' in row:
        row.pop('real_name', None)
    elif 'real_name' in row and 'Real_Name' not in row:
        row['Real_Name'] = row.pop('real_name')

    # If still missing, create it from host_name
    if 'Real_Name' not in row or pd.isna(row['Real_Name']):
        row['Real_Name'] = host_name

    # paper_name fallback
    if 'paper_name' not in row:
        row['paper_name'] = np.nan

    # Remove legacy column if present
    if 'host_name' in row:
        row.pop('host_name', None)

    # ---- FILTER: keep only paper_name H<=31 (inclusive) ----
    pn = row.get('paper_name', np.nan)
    hn = _hnum(pn)
    if hn is None or hn > 31:
        continue

    # Add target AFTER paper_name in final ordering (we'll enforce ordering later)
    row['dG'] = float(dG.loc[host_name])

    rows.append(row)

# ---------------- Build combined dataframe ----------------
all_data = pd.DataFrame(rows)

# Drop rows missing dG only (don't drop because pore columns etc might be NaN)
all_data = all_data.dropna(subset=['dG'])

# ---- FORCE final column order: Real_Name, paper_name, dG, rest ----
cols = list(all_data.columns)
ordered = []
for c in ['Real_Name', 'paper_name', 'dG']:
    if c in cols:
        ordered.append(c)
ordered += [c for c in cols if c not in ordered]
all_data = all_data[ordered]

# Write to CSV (IMPORTANT: index=False prevents the extra Real_Name column)
all_data.to_csv('all-host-props-nopore/all_host_props_training.csv', index=False)

print("Wrote: all-host-props-nopore/all_host_props_training.csv")
print("Columns:", list(all_data.columns))
print("Rows:", len(all_data))

