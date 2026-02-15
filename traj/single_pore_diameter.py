import sys
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
path='/afs/crc.nd.edu/user/m/mfarshad/Private/ML/hole2/exe/hole'

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('axes', labelsize='28')
rc('xtick', labelsize='28')
rc('ytick', labelsize='28')

warnings.filterwarnings('ignore')

folder_path = '/afs/crc.nd.edu/user/m/mfarshad/Private/ML/traj/'

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
subfoldername=sys.argv[1]
run_files = glob.glob(os.path.join(folder_path, subfoldername, 'host_*.mol2'))
run_names = [os.path.splitext(os.path.basename(file))[0][5:] for file in run_files]

# Get the current directory
current_dir = os.path.basename(os.getcwd())
# Use the current directory as the run_1 variable
run_1 = current_dir

# for run_1 in run_names:
pp = PdfPages('pore_distribution_' + str(run_1) + '.pdf')
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

pdb = glob.glob(os.path.join(folder_path, subfoldername, f'host_*.mol2'))
trr = glob.glob(os.path.join(folder_path, subfoldername, f'host_*.trr'))
u = mda.Universe(pdb)

ha = hole2.HoleAnalysis(u, select='all',
                        cpoint='center_of_geometry',
                        executable=path,
                        )
ha.run(random_seed=31415)

radii, edges = ha.bin_radii(bins=100, range=None)
means, edges = ha.histogram_radii(bins=100, range=None, aggregator=np.mean)
midpoints = 0.5 * (edges[1:] + edges[:-1])
midpoints = midpoints - midpoints[int(len(midpoints) / 2)]

# pp = PdfPages('pore_distribution_' + str(run_1) + '.pdf')
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].plot(midpoints, means, '-o', color='k')
axes[0].set_xlabel(r'$\mathbf{\textbf{Pore\ coordinate}\ \zeta\ (\AA)}$', labelpad=10, fontsize=20)
axes[0].set_ylabel(r'$\mathbf{\textbf{Mean\ pore\ radius}\ (\AA)}$', labelpad=10, fontsize=20)

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

print('average diameter middle radii = ', np.mean(diameter_middle_all))
print('average diameter all radii = ', np.mean(diameter_all))

output_file_path = f'min_diameter_profile_{run_1}.csv'
np.savetxt(output_file_path, [min(means)], delimiter=',', header='Min diameter_profile (Å)', comments='')

output_file_path = f'diameter_profile_{run_1}.csv'
np.savetxt(output_file_path, [means], delimiter=',', header='Diameter profile (Å)', comments='')

output_file_path = f'average_diameter_{run_1}.csv'
np.savetxt(output_file_path, [np.mean(diameter_all)], delimiter=',', header='Average Diameter (Å)', comments='')

output_file_path = f'diameter_distribution_{run_1}.csv'
np.savetxt(output_file_path, [diameter_all], delimiter=',', header='Diameter distribution(Å)', comments='')

axes[1].plot(midpoints, hist_middle_all, '-o', color='k')
axes[1].plot(midpoints, hist_all, '-o', color='k')

axes[1].set_xlabel(r'$\mathbf{\textbf{Pore\ distribution}\ (\AA)}$', labelpad=10, fontsize=20)
axes[1].set_ylabel(r'$\textbf{Probability}$', labelpad=16, fontsize=20)

ha.create_vmd_surface(filename= str(run_1)+'.vmd')

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
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()


