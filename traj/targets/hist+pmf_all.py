import numpy as np
import glob
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
# importing Statistics module
import statistics
from scipy.signal import savgol_filter

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize='28')
rc('xtick', labelsize='28')
rc('ytick', labelsize='28')

run_1 = 1
pp = PdfPages('hist+pmf' + str(run_1) + '_{real_name}.pdf')
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(hspace=0.0)

# --- NEW: minor ticks + formatter like GP code ---
for ax in axes:
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(formatter)
# -----------------------------------------------

# Placeholder for real_name
fname1 = glob.glob(
    '/afs/crc/group/whitmer/Data-MF-{number}/amber/{real_name}/Umbrella_sampling/'
    'umbrella/run_long_' + str(run_1) + '/histo.xvg'
)

x = [np.loadtxt(f, usecols=(0,), skiprows=18) for f in fname1]

for i in range(0, {length}):
    y = [np.loadtxt(f, usecols=(i + 1,), skiprows=18) for f in fname1]
    y = masked_y = np.ma.masked_where(y[0] == 0, y[0])
    # --- slight style tweak: consistent line width ---
    axes[0].plot(x[0], y, '-', lw=1.5)

axes[0].set_xticks(np.arange(0, max(x[0]) + 0.2, 0.2))
axes[0].set_ylabel(r'$\textbf{Counts}$', labelpad=20, fontsize=28)
axes[1].set_xlim(0.0, 2.5)
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
axes[0].set_xticklabels([])

fname1 = glob.glob(
    '/afs/crc/group/whitmer/Data-MF-{number}/amber/{real_name}/Umbrella_sampling/'
    'umbrella/run_long_' + str(run_1) + '/profile.xvg'
)
y = [np.loadtxt(f, usecols=(1,), skiprows=18) for f in fname1]
x = [np.loadtxt(f, usecols=(0,), skiprows=18) for f in fname1]

# Determine the minimum value
y_min = np.min(y[0])
# Shift the y-values
y = y - y_min

axes[1].plot(x[0], y[0], '-', c='b', lw=1.8)
# axes.fill_between(X_, Y_ - statistics.stdev(Y_), Y_ + statistics.stdev(Y_), alpha=0.35)
# axes[1].fill_between(x[0], y[0] - statistics.stdev(y[0]), y[0] + statistics.stdev(y[0]), alpha=0.35)

axes[1].set_xticks(np.arange(0, max(x[0]) + 0.2, 0.2))
axes[1].set_xlabel(r'$\mathbf{\Delta r\ \textbf{(\textit{nm})}}$', fontsize=28)
axes[1].set_ylabel(r'$\textbf{PMF (\textit{kcal/mol})}$', labelpad=16, fontsize=28)

y = y[0]
x = x[0]

# Apply a smoothing filter to the data
y_smoothed = savgol_filter(y, 31, 3)

# Find the global minima and maxima
min_idx = None
min_y = float('inf')  # Track the deepest valley
thresholds = [0.4, 0.6, 0.8, 1.0, 1.2]

# Step 1: Find the absolute minimum below 1.2 (prioritizing lower thresholds)
for threshold in thresholds:
    for i in range(1, len(y) - 1):
        if x[i] < threshold:
            if y[i] < y[i - 1] and y[i] < y[i + 1]:  # Local minimum check
                if y[i] < min_y:  # Only update if it's the deepest so far
                    min_y = y[i]
                    min_idx = i

# Step 2: Find the absolute maximum AFTER min_idx (but below x < 2.0)
max_idx = None
max_y = -float('inf')  # Track the highest peak

if min_idx is not None:  # Only search for max if a min was found
    for i in range(min_idx + 1, len(y) - 1):  # Start AFTER min_idx
        if x[i] > 1.0 and x[i] < 2.2:
            if y[i] > y[i - 1] and y[i] > y[i + 1]:  # Local maximum check
                if y[i] > max_y:  # Only update if it's the highest so far
                    max_y = y[i]
                    max_idx = i

diff = y[max_idx] - y[min_idx]
text_x = np.max(x[min_idx:max_idx]) * 1.0
text_y = ((y[min_idx] + y[max_idx]) / 2) * 2.5
min_x = x[min_idx]
max_x = x[max_idx]

axes[1].plot(x[min_idx], y[min_idx], 'co', markersize=18)
axes[1].plot(x[max_idx], y[max_idx], 'mo', markersize=18)
axes[1].text(
    text_x,
    text_y,
    r'$\Delta G^\ddagger = {:.2f}\ \mathrm{{kcal/mol}}$'.format(diff),
    fontsize=32,
    ha='center',
    va='bottom'
)
# Draw horizontal and vertical lines to connect the minimum and maximum points
axes[1].plot([min(x), max(x)], [y[min_idx], y[min_idx]], 'k--')
axes[1].plot([max_x, max_x], [y[min_idx], y[max_idx]], 'k--')

# axes[1].legend(
#     frameon=False,
#     borderpad=0.1,
#     labelspacing=0.2,
#     columnspacing=0.2,
#     borderaxespad=0.4,
#     handletextpad=0.4,
#     fontsize='22',
#     loc=1,
#     handlelength=0.5
# )

axes[1].set_xlim(0.0, 2.5)
axes[1].xaxis.set_ticks(np.arange(0.4, 2.7, 0.4))

# Reverse the y-values
y = y[max_idx] - y

first_positive_index = np.argmax(y > 0)

# Constants
R = 1.98720425864083  # Gas constant
T = 300               # Temperature in Kelvin
conversion = 4.1839953808691

# Calculate the sum of the exponential function
bound = np.sum(np.exp(y[0:max_idx] / (R * T / 1000)))
unbound = np.sum(np.exp(y[max_idx + 1:] / (R * T / 1000)))
constant = bound / unbound
standard_constant = constant
print(standard_constant)
free_energy_cal_ns = -(R * T * np.log(standard_constant)) / 1000
free_energy_J_ns = conversion * free_energy_cal_ns

# with correction
bound = np.sum(
    np.exp(y[0:max_idx] / (R * T / 1000)) *
    ((1) / (4 / 3 * np.pi * x[0:max_idx]**3))
)
unbound = np.sum(
    np.exp(y[max_idx + 1:] / (R * T / 1000)) *
    ((1.661) / (4 / 3 * np.pi * x[max_idx + 1:]**3)**2)
)
constant = bound / unbound
standard_constant = constant
print(standard_constant)
free_energy_cal_mo = -(R * T * np.log(standard_constant)) / 1000
free_energy_J_mo = conversion * free_energy_cal_mo

bound = np.sum(np.exp(y[0:max_idx] / (R * T / 1000)))
unbound = np.sum(np.exp(y[max_idx + 1:] / (R * T / 1000)))
constant = bound / unbound
standard_constant = constant * (4 / 3 * np.pi * max(x)**3) / (1.661)
print(standard_constant)
free_energy_cal_lit = -(R * T * np.log(standard_constant)) / 1000
free_energy_J_lit = conversion * free_energy_cal_lit

print("Delta G non-standard=", free_energy_cal_ns, ' kcal/mol')
print("Delta G non-standard=", free_energy_J_ns, ' kJ/mol')

print("Delta G standard-mohsen=", free_energy_cal_mo, ' kcal/mol')
print("Delta G srandard-mohsen=", free_energy_J_mo, ' kJ/mol')

print("Delta G standard-litrature=", free_energy_cal_lit, ' kcal/mol')
print("Delta G standard-litrature=", free_energy_J_lit, ' kJ/mol')

print("{real_name}", free_energy_cal_ns, free_energy_cal_mo, free_energy_cal_lit)

plt.tight_layout()
pp.savefig(fig, bbox_inches='tight')
pp.close()
