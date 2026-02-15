#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

# Pretty printing for numpy
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

# Scientific notation formatter for y-axis
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # always scientific

# LaTeX-style fonts
rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize='28')
rc('xtick', labelsize='28')
rc('ytick', labelsize='28')

run_1 = 1

# Tail trimming (MINIMAL CHANGE)
n_tail_drop = 10

# PDF file name: {real_name} will be replaced by sed in the bash script
pp = PdfPages("hist+pmf" + str(run_1) + "_{real_name}.pdf")

# ---------- helper to load bsResult.xvg (x, mean, std) ----------
def load_bs_xvg(bs_file):
    """
    Load x, mean, std from a GROMACS bsResult.xvg file (bsResult.xvg).
    Skips lines starting with '@' or '#' and any non-numeric lines.
    """
    numeric_lines = []
    with open(bs_file, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s[0] in ('#', '@'):
                continue
            parts = s.split()
            try:
                _ = [float(p) for p in parts]
            except ValueError:
                continue
            numeric_lines.append(s)

    if not numeric_lines:
        raise ValueError(f"No numeric data found in {bs_file}")

    data_str = "\n".join(numeric_lines)
    data = np.loadtxt(io.StringIO(data_str))
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns (x, mean, std) in {bs_file}, "
            f"got {data.shape[1]}"
        )
    x_raw = data[:, 0]
    y_mean_raw = data[:, 1]  # bootstrap average PMF
    y_std_raw = data[:, 2]   # bootstrap std
    return x_raw, y_mean_raw, y_std_raw
# ---------------------------------------------------------------

# ---- single-panel PMF figure ----
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

# minor ticks + formatter
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_major_formatter(formatter)

# Path to bsResult.xvg
bs_file = "/afs/crc/group/whitmer/Data-MF-{number}/amber/{real_name}/Umbrella_sampling/umbrella/run_long_1/bsResult.xvg"
print("Reading:", bs_file)

# x_raw, mean_raw, std_raw
x_raw, y_mean_raw, y_std_raw = load_bs_xvg(bs_file)

# ========== MINIMAL CHANGE: APPLY n_tail_drop ==========
if n_tail_drop > 0:
    x = x_raw[:-n_tail_drop]
    y_mean = y_mean_raw[:-n_tail_drop]
    y_std = y_std_raw[:-n_tail_drop]
else:
    x = x_raw
    y_mean = y_mean_raw
    y_std = y_std_raw

# Shift mean PMF after truncation
y_min = np.min(y_mean)
y_full = y_mean - y_min
y_err_full = y_std.copy()
# ========================================================

# ================== Find min, barrier, and ΔG‡ ==================
min_idx = None
min_y = float('inf')
thresholds = [0.4, 0.6, 0.8, 1.0, 1.2]

# Step 1: find min
for threshold in thresholds:
    for i in range(1, len(y_full) - 1):
        if x[i] < threshold:
            if y_full[i] < y_full[i - 1] and y_full[i] < y_full[i + 1]:
                if y_full[i] < min_y:
                    min_y = y_full[i]
                    min_idx = i

# Step 2: find barrier
max_idx = None
max_y = -float('inf')
if min_idx is not None:
    for i in range(min_idx + 1, len(y_full) - 1):
        if 1.0 < x[i] < 2.2:
            if y_full[i] > y_full[i - 1] and y_full[i] > y_full[i + 1]:
                if y_full[i] > max_y:
                    max_y = y_full[i]
                    max_idx = i

if min_idx is None or max_idx is None:
    raise RuntimeError("Could not find suitable min/max for barrier detection.")

diff = y_full[max_idx] - y_full[min_idx]

# ---- Plot ----
ax.plot(x, y_full, '-', lw=3, label=r'{real_name}')
ax.fill_between(x, y_full - y_err_full, y_full + y_err_full, alpha=0.5)

ax.plot(x[min_idx], y_full[min_idx], 'co', markersize=18)
ax.plot(x[max_idx], y_full[max_idx], 'mo', markersize=18)
ax.plot([x.min(), x.max()], [y_full[min_idx], y_full[min_idx]], 'k--')
ax.plot([x[max_idx], x[max_idx]], [y_full[min_idx], y_full[max_idx]], 'k--')

ax.set_title(
    r"{real_name}: $\Delta G^\ddagger = %.2f\;(\mathrm{kcal\cdot mol^{-1}})$" % diff,
    fontsize=22
)

ax.set_xlim(0.0, 2.5)
ax.set_ylim(-0.5, 4.5)
ax.set_xticks(np.arange(0.4, x.max() + 1e-6, 0.4))
ax.set_xlabel(r"$\Delta r\;(\mathrm{nm})$", fontsize=28)
ax.set_ylabel(r"PMF $(\mathrm{kcal\cdot mol^{-1}})$", labelpad=16, fontsize=28)

# ================== Free energy ==================
y_rev = y_full[max_idx] - y_full

R = 1.98720425864083
T = 300.0
conversion = 4.1839953808691

bound = np.sum(np.exp(y_rev[0:max_idx] / (R * T / 1000.0)))
unbound = np.sum(np.exp(y_rev[max_idx + 1:] / (R * T / 1000.0)))
constant = bound / unbound
free_energy_cal_ns = -(R * T * np.log(constant)) / 1000.0

bound = np.sum(
    np.exp(y_rev[0:max_idx] / (R * T / 1000.0)) *
    (1.0 / (4.0 / 3.0 * np.pi * x[0:max_idx]**3))
)
unbound = np.sum(
    np.exp(y_rev[max_idx + 1:] / (R * T / 1000.0)) *
    (1.661 / (4.0 / 3.0 * np.pi * x[max_idx + 1:]**3)**2)
)
constant = bound / unbound
free_energy_cal_mo = -(R * T * np.log(constant)) / 1000.0

bound = np.sum(np.exp(y_rev[0:max_idx] / (R * T / 1000.0)))
unbound = np.sum(np.exp(y_rev[max_idx + 1:] / (R * T / 1000.0)))
constant = bound / unbound
constant = constant * (4.0/3.0 * np.pi * x.max()**3) / 1.661
free_energy_cal_lit = -(R * T * np.log(constant)) / 1000.0

print("Delta G non-standard=", free_energy_cal_ns, ' kcal/mol')
print("Delta G standard-mohsen=", free_energy_cal_mo, ' kcal/mol')
print("Delta G standard-literature=", free_energy_cal_lit, ' kcal/mol')

# captured by bash tail -n 1
print("{real_name}", free_energy_cal_ns, free_energy_cal_mo, free_energy_cal_lit)

plt.tight_layout()
pp.savefig(fig, bbox_inches='tight')
pp.close()

