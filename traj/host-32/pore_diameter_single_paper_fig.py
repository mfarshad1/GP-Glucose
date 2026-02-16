import os
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PDB_HOLE
from MDAnalysis.analysis import hole2
import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
import glob
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
# importing Statistics module
import statistics
from scipy.signal import savgol_filter
import warnings
# suppress some MDAnalysis warnings when writing PDB files
warnings.filterwarnings('ignore')

path = '/afs/crc.nd.edu/user/m/mfarshad/Private/ML/hole2/exe/hole'

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

# ==== Make style consistent with other paper figures ====
rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize=16)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)

warnings.filterwarnings('ignore')

folder_path = ''  # '/afs/crc.nd.edu/user/m/mfarshad/Private/ML/traj/pore_diameter/'

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

# use plain numbers on x-axis
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
formatter.set_powerlimits((-1, 1))

run_files = glob.glob(os.path.join(folder_path, 'host_*.mol2'))
run_names = [os.path.splitext(os.path.basename(file))[0][5:] for file in run_files]

# Get the current directory
current_dir = os.path.basename(os.getcwd())
print('currently running: ', current_dir)
# Use the current directory as the run_1 variable
run_1 = current_dir.split('-')[1:]
run_1 = '-'.join(run_1)

pp = PdfPages(f'pore_distribution_' + str(run_1) + '.pdf')
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

pdb = os.path.join(folder_path, f'host_{run_1}.mol2')
trr = os.path.join(folder_path, f'host_{run_1}.trr')
u = mda.Universe(pdb, trr)

ha = hole2.HoleAnalysis(u, select='all',
                        cpoint='center_of_geometry',
                        executable=path,
                        )
ha.run(random_seed=31415)

radii, edges = ha.bin_radii(bins=100, range=None)
means, edges = ha.histogram_radii(bins=100, range=None, aggregator=np.mean)
midpoints = 0.5 * (edges[1:] + edges[:-1])
midpoints = midpoints - midpoints[int(len(midpoints) / 2)]

# ==== Smooth the left-panel profile ====
if len(means) >= 7:
    # choose an odd window size, not too large
    win = min(21, len(means) // 2 * 2 - 1)
    smooth_means = savgol_filter(means, window_length=win, polyorder=3)
else:
    smooth_means = means.copy()

# Optionally show original points lightly (comment out if not desired)
axes[0].plot(midpoints, means, 'o', color='0.6', markersize=3)
axes[0].plot(midpoints, smooth_means, '-', color='k', linewidth=2)

axes[0].set_xlabel(r'$\mathbf{\textbf{Pore\ coordinate}\ \zeta\ (\AA)}$', labelpad=8)
axes[0].set_ylabel(r'$\mathbf{\textbf{Mean\ pore\ radius}\ (\AA)}$', labelpad=8)

diameter_middle_all = []
diameter_all = []
for i in range(0, len(ha.results.profiles)):
    if len(ha.results.profiles[i]) == 0:
        continue
    middle_ndx = int(len(ha.results.profiles[i]) / 2)
    diameter_middle = ha.results.profiles[i][middle_ndx][1]
    diameter_middle_all = np.append(diameter_middle_all, diameter_middle)
    for j in range(0, len(ha.results.profiles[i])):
        diameter = ha.results.profiles[i][j][1]
        diameter_all = np.append(diameter_all, diameter)

# Normalized histograms
hist_middle_all, edges = np.histogram(diameter_middle_all)
hist_middle_all = hist_middle_all / np.sum(hist_middle_all)

hist_all, edges = np.histogram(diameter_all)
hist_all = hist_all / np.sum(hist_all)

midpoints = 0.5 * (edges[1:] + edges[:-1])

print('average diameter middle radii = ', np.mean(diameter_middle_all))
print('average diameter all radii = ', np.mean(diameter_all))
diameter_std = np.std(diameter_all)          # population std
# or, if you prefer NaN-safe:
# diameter_std = np.nanstd(diameter_all)

print('std diameter all radii = ', diameter_std)

# ==== CSV outputs (unchanged) ====
output_file_path = f'min_diameter_profile_{run_1}.csv'
np.savetxt(output_file_path, [min(means)], delimiter=',',
           header='Min diameter_profile (Å)', comments='')

output_file_path = f'diameter_profile_{run_1}.csv'
np.savetxt(output_file_path, [means], delimiter=',',
           header='Diameter profile (Å)', comments='')

output_file_path = f'average_diameter_{run_1}.csv'
np.savetxt(output_file_path, [np.mean(diameter_all)], delimiter=',',
           header='Average Diameter (Å)', comments='')

output_file_path = f'diameter_distribution_{run_1}.csv'
np.savetxt(output_file_path, [diameter_all], delimiter=',',
           header='Diameter distribution(Å)', comments='')

# ==== Right panel: midpoint + full distribution (minor peak visible) ====
axes[1].plot(midpoints, hist_middle_all, '-o', color='k',
             linewidth=2, markersize=4, label=r'Midpoint')
axes[1].plot(midpoints, hist_all, '--', color='0.5',
             linewidth=2, label=r'All positions')

axes[1].set_xlabel(r'$\mathbf{\textbf{Pore\ diameter}\ (\AA)}$', labelpad=8)
axes[1].set_ylabel(r'$\textbf{Probability}$', labelpad=12)

axes[1].legend(frameon=False, fontsize=10, handlelength=2)

# Ticks / style tweaks for both panels
for ax in axes:
    ax.tick_params(direction='in', length=6, width=1)
    ax.xaxis.set_major_formatter(formatter)
    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in', length=3, width=0.8)
# ha.create_vmd_surface(filename=str(run_1) + '.vmd')

hole_path = '/afs/crc.nd.edu/user/m/mfarshad/Private/ML/'  # Replace with the actual path

all_names = os.listdir(hole_path)

# Exclude "hole2" from the list
exclude_names = ["hole2"]

# Filter the names that start with "hole" but are not in the exclude list
hole_names = [name for name in all_names if name.startswith("hole") and name not in exclude_names]

# Remove files
for file_name in hole_names:
    file_path = os.path.join(hole_path, file_name)
    os.remove(file_path)

plt.tight_layout()
pp.savefig(fig, bbox_inches='tight')
pp.close()
