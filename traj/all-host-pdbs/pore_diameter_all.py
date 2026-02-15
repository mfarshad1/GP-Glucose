import os
import sys
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PDB_HOLE
from MDAnalysis.analysis import hole2
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
import statistics
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')

# Set the path to the hole executable
path = '/afs/crc.nd.edu/user/m/mfarshad/Private/ML/hole2/exe/hole'

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python pore_diameter_all.py folder_name")
    sys.exit(1)

# Get the folder name from command-line argument
folder_name = sys.argv[1]

# Construct the path to the .gro and .trr files
parent_directory = "/afs/crc.nd.edu/user/m/mfarshad/Private/ML/traj"
current_directory = os.path.join(parent_directory, folder_name)
gro_file = os.path.join(current_directory, f'host_{folder_name}.gro')
trr_file = os.path.join(current_directory, f'host_{folder_name}.trr')

# Create a PdfPages object for saving plots
pp = PdfPages('pore_distribution_' + str(folder_name) + '.pdf')

# Create subplots for plotting
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Construct the path to the .gro and .trr files
gro_file = os.path.join(current_directory, f'host_{folder_name}.gro')
trr_file = os.path.join(current_directory, f'host_{folder_name}.trr')

# Load the trajectory using MDAnalysis
u = mda.Universe(gro_file, trr_file)

# Perform hole analysis
ha = hole2.HoleAnalysis(u, select='all',
                        cpoint='center_of_geometry',
                        executable=path)

ha.run(random_seed=31415)

# Calculate pore radius distribution
radii, edges = ha.bin_radii(bins=100, range=None)
means, edges = ha.histogram_radii(bins=100, range=None, aggregator=np.mean)
midpoints = 0.5 * (edges[1:] + edges[:-1])
midpoints = midpoints - midpoints[int(len(midpoints) / 2)]

# Plot mean pore radius
axes[0].plot(midpoints, means, '-o', color='k')
axes[0].set_xlabel(r'$\mathbf{\textbf{Pore\ coordinate}\ \zeta\ (\AA)}$', labelpad=10, fontsize=20)
axes[0].set_ylabel(r'$\mathbf{\textbf{Mean\ pore\ radius}\ (\AA)}$', labelpad=10, fontsize=20)

# Calculate pore diameter distribution
diameter_middle_all = []
diameter_all = []
for i in range(0, len(ha.results.profiles)):
    middle_ndx = int(len(ha.results.profiles[i]) / 2)
    if len(ha.results.profiles[i]) == 0:
        continue
    diameter_middle = ha.results.profiles[i][middle_ndx][1]
    diameter_middle_all = np.append(diameter_middle_all, diameter_middle)
    for j in range(0, len(ha.results.profiles[i])):
        diameter = ha.results.profiles[i][j][1]
        diameter_all = np.append(diameter_all, diameter)

hist_middle_all, edges = np.histogram(diameter_middle_all)
hist_middle_all = hist_middle_all / np.sum(hist_middle_all)

hist_all, edges = np.histogram(diameter_all)
hist_all = hist_all / np.sum(hist_all)

midpoints = 0.5 * (edges[1:] + edges[:-1])

output_file_path = f'min_diameter_profile_{folder_name}.csv'
np.savetxt(output_file_path, [min(means)], delimiter=',', header='Average Diameter (Å)', comments='')

output_file_path = f'diameter_profile_{folder_name}.csv'
np.savetxt(output_file_path, [means], delimiter=',', header='Average Diameter (Å)', comments='')

output_file_path = f'average_diameter_{folder_name}.csv'
np.savetxt(output_file_path, [np.mean(diameter_all)], delimiter=',', header='Average Diameter (Å)', comments='')

axes[1].plot(midpoints, hist_middle_all, '-o', color='k')
axes[1].plot(midpoints, hist_all, '-o', color='k')

axes[1].set_xlabel(r'$\mathbf{\text{Pore\ distribution}\ (\AA)}$', labelpad=10, fontsize=20)
axes[1].set_ylabel(r'$\textbf{Probability}$', labelpad=16, fontsize=20)

# Create VMD surface
# ha.create_vmd_surface(filename=f'{folder_name}.vmd')

# Remove files from the directory
for file_name in os.listdir():
    if file_name.startswith("hole") and file_name != "hole2":
        os.remove(file_name)

plt.tight_layout()
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()
