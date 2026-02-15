#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Histogram (counts) + bootstrap PMF (mean ± std) for a single host.

- Top panel: histograms from histo.xvg.
- Bottom panel: PMF from bsResult.xvg (mean ± std, shifted so min = 0).
- Uses similar formatting to the 2x2 bootstrap PMF script:
  LaTeX fonts, bold-style labels, kcal·mol^{-1}, etc.
"""

import io
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

# ================= General settings =================
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

# Formatter for PMF axis
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # always scientific for PMF axis

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize='28')
rc('xtick', labelsize='28')
rc('ytick', labelsize='28')

run_1 = 1

# ------- SET THESE FOR THE HOST YOU WANT -------
number = "07"        # Data-MF-XX
real_name = "95"     # host index (e.g. "20")
# -----------------------------------------------

# Tail truncation for PMF (same idea as 2x2 script)
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

# ---------- helper to load histo.xvg (counts per window) -------
def load_histo_xvg(histo_file):
    """
    Load reaction coordinate + histograms from histo.xvg.
    Skips '@' and '#' lines and any non-numeric lines.
    Returns x array and a 2D array of counts (n_points × n_windows).
    """
    numeric_lines = []
    with open(histo_file, 'r') as f:
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
        raise ValueError(f"No numeric data found in {histo_file}")

    data_str = "\n".join(numeric_lines)
    data = np.loadtxt(io.StringIO(data_str))
    if data.ndim == 1:
        data = data[np.newaxis, :]

    x = data[:, 0]
    y_all = data[:, 1:]  # each column is one umbrella window
    return x, y_all
# ---------------------------------------------------------------

# ---- two-panel figure: top = counts, bottom = PMF ----
fig, axes = plt.subplots(2, 1, figsize=(7, 8))
plt.subplots_adjust(hspace=0.15)

ax_hist = axes[0]
ax_pmf  = axes[1]

# minor ticks on both
for ax in axes:
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# === FORMATTERS ===
# PMF y-axis: use main formatter
formatter.set_useOffset(False)
ax_pmf.yaxis.set_major_formatter(formatter)

# Histogram y-axis: its own scientific formatter
hist_formatter = ticker.ScalarFormatter(useMathText=True)
hist_formatter.set_scientific(True)
hist_formatter.set_powerlimits((0, 0))
hist_formatter.set_useOffset(False)
ax_hist.yaxis.set_major_formatter(hist_formatter)

# ----------------- Top panel: histograms (histo.xvg) -----------------
histo_file = (
    f"/afs/crc/group/whitmer/Data-MF-{number}/amber/{real_name}/Umbrella_sampling/"
    f"umbrella/run_long_{run_1}/histo.xvg"
)
print("Reading histogram:", histo_file)

x_hist, y_hist_all = load_histo_xvg(histo_file)

# Plot each window's histogram (mask zeros)
for j in range(y_hist_all.shape[1]):
    y = y_hist_all[:, j]
    y_masked = np.ma.masked_where(y == 0.0, y)
    ax_hist.plot(x_hist, y_masked, '-', lw=1.5)

ax_hist.set_xlim(0.0, 2.5)
ax_hist.set_xticks(np.arange(0.4, 2.7, 0.4))
ax_hist.set_xticklabels([])  # no x-labels on top panel
ax_hist.set_ylabel(r"$\mathbf{Counts}$", labelpad=16, fontsize=28)

# ----------------- Bottom panel: PMF (bsResult.xvg) ------------------
bs_file = (
    f"/afs/crc/group/whitmer/Data-MF-{number}/amber/{real_name}/Umbrella_sampling/"
    f"umbrella/run_long_{run_1}/bsResult.xvg"
)
print("Reading PMF:", bs_file)

x_raw, y_mean_raw, y_std_raw = load_bs_xvg(bs_file)

# Drop tail bins if requested
if n_tail_drop > 0:
    x = x_raw[:-n_tail_drop]
    y_mean = y_mean_raw[:-n_tail_drop]
    y_std  = y_std_raw[:-n_tail_drop]
else:
    x = x_raw
    y_mean = y_mean_raw
    y_std  = y_std_raw

# Shift mean PMF so minimum = 0
y_min = np.min(y_mean)
y_full = y_mean - y_min
y_err_full = y_std.copy()

# ===== barrier detection on truncated y_full (same logic as 2x2) =====
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

# ---- plotting PMF: truncated profile, mean ± std ----
x_plot = x
y_plot = y_full
y_err_plot = y_err_full

ax_pmf.plot(x_plot, y_plot, '-', lw=3, label=rf"H{real_name}")
ax_pmf.fill_between(x_plot, y_plot - y_err_plot, y_plot + y_err_plot, alpha=0.5)

# Markers + lines
ax_pmf.plot(x[min_idx], y_full[min_idx], 'co', markersize=18)
ax_pmf.plot(x[max_idx], y_full[max_idx], 'mo', markersize=18)
ax_pmf.plot([x.min(), x.max()], [y_full[min_idx], y_full[min_idx]], 'k--')
ax_pmf.plot([x[max_idx], x[max_idx]], [y_full[min_idx], y_full[max_idx]], 'k--')

# Title with ΔG‡ (kcal·mol^{-1})
ax_pmf.set_title(
    rf"H20: $\Delta G^\ddagger = {diff:.2f}\;(\mathrm{{kcal\cdot mol^{-1}}})$",
    fontsize=22
)

# Axes formatting for PMF
ax_hist.set_xlim(0.0, 2.5)
ax_pmf.set_xlim(0.0, 2.5)
ax_pmf.set_ylim(-0.5, 4.5)
ax_pmf.set_xticks(np.arange(0.4, x.max() + 1e-6, 0.4))

# >>> This is where I changed the labels to match your 2x2 style <<<
ax_pmf.set_xlabel(
    r'$\mathbf{\Delta r\ \textbf{(\textit{nm})}}$',
    fontsize=28
)
ax_pmf.set_ylabel(
    r'$\textbf{PMF (\textit{kcal$\cdot$mol$^{-1}$})}$',
    labelpad=16,
    fontsize=28
)

# ================== Free energy calculations on y_full ==================
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
