#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMF plots for several hosts (2x2), using bootstrap WHAM output.

- Reads bsResult.xvg (x, mean, std) from WHAM with -nBootstrap.
- Shifts mean PMF so minimum = 0.
- Plots PMF with a shaded confidence interval (mean ± std).
- Finds ΔG^‡ from bootstrap-mean PMF and shows it in the title.
- Keeps plotting style as close as possible to the original code.
"""

import os
import io
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter

# ================= General settings =================
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # always scientific

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize='28')
rc('xtick', labelsize='28')
rc('ytick', labelsize='28')

# ------------------- USER INPUTS --------------------
run_1 = 1  # umbrella run index

# molecule -> legend label
# (edit the host IDs here as needed)
mol_info = [
    ("12",  "H1"),
    ("32",  "H8"),
    ("87",  "H19"),
    ("174", "H25"),
]

# Data-MF folders to search, in order of preference
data_folders = ["09", "08", "07", "06", "05", "04", "03", "02", "01"]

# How many last bins to drop from PMF (to remove strange tail)
n_tail_drop = 10
# ---------------------------------------------------


def load_bs_xvg(bs_file):
    """
    Load x, mean, std from a GROMACS xvg file with headers.

    Keeps only purely numeric data lines and ignores headers/comments.
    Assumes the first three columns are: x, mean, std.
    """
    numeric_lines = []
    with open(bs_file, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Skip obvious header/comment lines
            if s[0] in ('#', '@'):
                continue
            parts = s.split()
            # Keep only lines where *all* tokens are floats
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
    y_mean_raw = data[:, 1]
    y_std_raw = data[:, 2]
    return x_raw, y_mean_raw, y_std_raw


pp = PdfPages(f"pmf_bootstrap_2x2_run{run_1}.pdf")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.25, wspace=0.25)

axes_flat = axes.flatten()

# --- minor ticks + formatter on all panels ---
for ax in axes_flat:
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(formatter)

# Constants used for free‐energy calculations
R = 1.98720425864083   # cal/(mol·K)
T = 300.0              # K
conversion = 4.1839953808691  # kJ/mol per kcal/mol

for ax, (real_name, legend_label) in zip(axes_flat, mol_info):

    # ---------- find the correct Data-MF-* folder ----------
    bs_file = None
    base_dir = None
    for number in data_folders:
        candidate_base = (
            f"/afs/crc/group/whitmer/Data-MF-{number}/amber/"
            f"{real_name}/Umbrella_sampling/umbrella/run_long_{run_1}"
        )
        candidate_bs = os.path.join(candidate_base, "bsResult.xvg")
        if os.path.exists(candidate_bs):
            base_dir = candidate_base
            bs_file = candidate_bs
            print(f"Using {candidate_bs} for molecule {real_name} ({legend_label})")
            break

    if bs_file is None:
        raise FileNotFoundError(
            f"bsResult.xvg not found for molecule {real_name} ({legend_label}) "
            f"in Data-MF-{data_folders}."
        )

    # =================== Bootstrap PMF (mean + std) ===================
    x_raw, y_mean_raw, y_std_raw = load_bs_xvg(bs_file)

    # Drop last few bins to remove strange tail
    if n_tail_drop > 0:
        x = x_raw[:-n_tail_drop]
        y_mean = y_mean_raw[:-n_tail_drop]
        y_std = y_std_raw[:-n_tail_drop]
    else:
        x = x_raw
        y_mean = y_mean_raw
        y_std = y_std_raw

    # Shift mean PMF so minimum = 0 (after truncation)
    y_min = y_mean.min()
    y = y_mean - y_min  # this is the PMF we will use for everything
    y_err = y_std.copy()  # std is unaffected by shifting the mean

    # Optional smoothing of mean (kept from original style)
    # if len(y) >= 31:
    #     y_smooth = savgol_filter(y, 31, 3)
    # else:
    #     y_smooth = y.copy()

    # Plot PMF line (smoothed) and shaded confidence interval (unsmoothed ± std)
    ax.plot(x, y, '-', lw=3, label=legend_label)
    ax.fill_between(x, y - y_err, y + y_err, alpha=0.5)

    # =================== Find min, barrier, and ΔG‡ on mean PMF ===================
    min_idx = None
    min_y_loc = float('inf')
    thresholds = [0.4, 0.6, 0.8, 1.0, 1.2]

    # Step 1: absolute minimum with x < 1.2, prioritizing lower thresholds
    for threshold in thresholds:
        for i in range(1, len(y) - 1):
            if x[i] < threshold:
                if y[i] < y[i - 1] and y[i] < y[i + 1]:
                    if y[i] < min_y_loc:
                        min_y_loc = y[i]
                        min_idx = i

    # Step 2: maximum after min_idx with 1.0 < x < 2.2
    max_idx = None
    max_y_loc = -float('inf')
    if min_idx is not None:
        for i in range(min_idx + 1, len(y) - 1):
            if 1.0 < x[i] < 2.2:
                if y[i] > y[i - 1] and y[i] > y[i + 1]:
                    if y[i] > max_y_loc:
                        max_y_loc = y[i]
                        max_idx = i

    if min_idx is None or max_idx is None:
        raise RuntimeError(
            f"Could not find suitable min/max for barrier detection for "
            f"molecule {real_name} ({legend_label})."
        )

    diff = y[max_idx] - y[min_idx]  # ΔG‡ in kcal/mol (since y is in kcal/mol)

    min_x = x[min_idx]
    max_x = x[max_idx]

    # Markers and dashed lines as in the original code
    ax.plot(x[min_idx], y[min_idx], 'co', markersize=18)
    ax.plot(x[max_idx], y[max_idx], 'mo', markersize=18)
    ax.plot([x.min(), x.max()], [y[min_idx], y[min_idx]], 'k--')
    ax.plot([max_x, max_x], [y[min_idx], y[max_idx]], 'k--')

    # Put ΔG‡ in the TITLE
    ax.set_title(
        rf'{legend_label}: $\Delta G^\ddagger = {diff:.2f}\ \mathrm{{kcal/mol}}$',
        fontsize=22
    )

    # =================== Binding free energies (using bootstrap-mean PMF) ===================
    # reverse PMF relative to barrier
    y_rev = y[max_idx] - y

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

    # Print summary for this host
    print(f"Molecule {real_name} ({legend_label}): ΔG^‡ = {diff:.3f} kcal/mol")
    print(f"  ΔG non-standard        = {free_energy_cal_ns:.3f} kcal/mol "
          f"({free_energy_J_ns:.3f} kJ/mol)")
    print(f"  ΔG standard-mohsen     = {free_energy_cal_mo:.3f} kcal/mol "
          f"({free_energy_J_mo:.3f} kJ/mol)")
    print(f"  ΔG standard-literature = {free_energy_cal_lit:.3f} kcal/mol "
          f"({free_energy_J_lit:.3f} kJ/mol)")
    print("--------------------------------------------------------")

    # Axes formatting for this panel
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks(np.arange(0.4, x.max() + 1e-6, 0.4))
    ax.set_xlabel(r'$\mathbf{\Delta r\ \textbf{(\textit{nm})}}$', fontsize=24)
    ax.set_ylabel(r'$\textbf{PMF (\textit{kcal/mol})}$', labelpad=10, fontsize=24)
    # ax.legend(frameon=False, loc='best')

plt.tight_layout()
pp.savefig(fig, bbox_inches='tight')
pp.close()
