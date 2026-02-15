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

# >>>>> SET THESE CORRECTLY FOR THIS HOST <<<<<
number = "08"        # e.g. "08"
real_name = "32"     # e.g. "32" for host-32
# >>>>> -------------------------------- <<<<<

# How many last bins to drop (same as 2×2 script)
n_tail_drop = 10

pp = PdfPages(f"hist+pmf{run_1}_{real_name}.pdf")

# ---------- helper to load bsResult.xvg (x, mean, std) ----------
def load_bs_xvg(bs_file):
    """
    Load x, mean, std from a GROMACS bsResult.xvg file.
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
bs_file = (
    f"/afs/crc/group/whitmer/Data-MF-{number}/amber/{real_name}/Umbrella_sampling/"
    f"umbrella/run_long_{run_1}/bsResult.xvg"
)
print("Reading:", bs_file)

# x_raw, mean_raw, std_raw from bsResult.xvg
x_raw, y_mean_raw, y_std_raw = load_bs_xvg(bs_file)

# ---- DROP LAST BINS (to match 2×2 script) ----
if n_tail_drop > 0:
    x = x_raw[:-n_tail_drop]
    y_mean = y_mean_raw[:-n_tail_drop]
    y_std = y_std_raw[:-n_tail_drop]
else:
    x = x_raw
    y_mean = y_mean_raw
    y_std = y_std_raw

# mean (2nd column, truncated) as reference PMF
y_min = np.min(y_mean)
y_full = y_mean - y_min       # shifted mean PMF (truncated)
y_err_full = y_std.copy()     # std (truncated)
# ------------------------------------------------

# ================== Find min, barrier, and ΔG‡ on y_full ==================
min_idx = None
min_y = float('inf')
thresholds = [0.4, 0.6, 0.8, 1.0, 1.2]

# Step 1: absolute minimum with x < 1.2
for threshold in thresholds:
    for i in range(1, len(y_full) - 1):
        if x[i] < threshold:
            if y_full[i] < y_full[i - 1] and y_full[i] < y_full[i + 1]:
                if y_full[i] < min_y:
                    min_y = y_full[i]
                    min_idx = i

# Step 2: maximum after min_idx with 1.0 < x < 2.2
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

# ---- plotting: use the truncated profile (no smoothing) ----
x_plot = x
y_plot = y_full
y_err_plot = y_err_full

ax.plot(x_plot, y_plot, '-', lw=3, label=rf'{real_name}')
ax.fill_between(x_plot, y_plot - y_err_plot, y_plot + y_err_plot, alpha=0.5)

# Text and markers (full-data positions in the truncated arrays)
text_x = np.max(x[min_idx:max_idx]) * 1.0
text_y = ((y_full[min_idx] + y_full[max_idx]) / 2) * 2.5

ax.plot(x[min_idx], y_full[min_idx], 'co', markersize=18)
ax.plot(x[max_idx], y_full[max_idx], 'mo', markersize=18)
ax.plot([x.min(), x.max()], [y_full[min_idx], y_full[min_idx]], 'k--')
ax.plot([x[max_idx], x[max_idx]], [y_full[min_idx], y_full[max_idx]], 'k--')

# ΔG‡ in title, with fixed unit so -1 is superscript
ax.set_title(
    rf"{real_name}: $\Delta G^\ddagger = {diff:.2f}\;(\mathrm{{kcal}}\cdot\mathrm{{mol}}^{{-1}})$",
    fontsize=22
)

# Axes formatting; labels in the same style, also fixed
ax.set_xlim(0.0, 2.5)
ax.set_ylim(-0.5, 4.5)
ax.set_xticks(np.arange(0.4, x.max() + 1e-6, 0.4))
ax.set_xlabel(r"$\Delta r\;(\mathrm{nm})$", fontsize=28)
ax.set_ylabel(r"PMF $(\mathrm{kcal}\cdot\mathrm{mol}^{-1})$", labelpad=16, fontsize=28)

# ================== Free energy calculations on y_full (truncated) ==================
y_rev = y_full[max_idx] - y_full

R = 1.98720425864083  # cal/mol/K
T = 300.0
conversion = 4.1839953808691  # kJ per kcal

# non-standard
bound = np.sum(np.exp(y_rev[0:max_idx] / (R * T / 1000.0)))
unbound = np.sum(np.exp(y_rev[max_idx + 1:] / (R * T / 1000.0)))
constant = bound / unbound
standard_constant = constant
free_energy_cal_ns = -(R * T * np.log(standard_constant)) / 1000.0
free_energy_J_ns = conversion * free_energy_cal_ns

# Mohsen correction
bound = np.sum(
    np.exp(y_rev[0:max_idx] / (R * T / 1000.0)) *
    (1.0 / (4.0 / 3.0 * np.pi * x[0:max_idx]**3))
)
unbound = np.sum(
    np.exp(y_rev[max_idx + 1:] / (R * T / 1000.0)) *
    (1.661 / (4.0 / 3.0 * np.pi * x[max_idx + 1:]**3)**2)
)
constant = bound / unbound
standard_constant = constant
free_energy_cal_mo = -(R * T * np.log(standard_constant)) / 1000.0
free_energy_J_mo = conversion * free_energy_cal_mo

# literature-style correction
bound = np.sum(np.exp(y_rev[0:max_idx] / (R * T / 1000.0)))
unbound = np.sum(np.exp(y_rev[max_idx + 1:] / (R * T / 1000.0)))
constant = bound / unbound
standard_constant = constant * (4.0 / 3.0 * np.pi * x.max()**3) / 1.661
free_energy_cal_lit = -(R * T * np.log(standard_constant)) / 1000.0
free_energy_J_lit = conversion * free_energy_cal_lit

print("Delta G non-standard=", free_energy_cal_ns, ' kcal/mol')
print("Delta G non-standard=", free_energy_J_ns, ' kJ/mol')
print("Delta G standard-mohsen=", free_energy_cal_mo, ' kcal/mol')
print("Delta G standard-mohsen=", free_energy_J_mo, ' kJ/mol')
print("Delta G standard-literature=", free_energy_cal_lit, ' kcal/mol')
print("Delta G standard-literature=", free_energy_J_lit, ' kJ/mol')
print(real_name, free_energy_cal_ns, free_energy_cal_mo, free_energy_cal_lit)

plt.tight_layout()
pp.savefig(fig, bbox_inches='tight')
pp.close()
