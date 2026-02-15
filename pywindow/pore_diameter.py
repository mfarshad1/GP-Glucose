import pywindow as pw
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import json

traj = pw.trajectory.XYZ("/afs/crc/group/whitmer/Data-MF-05/ML/traj/host_asb1.xyz")

traj.no_of_frames

frame_0 = traj.get_frames(0)
frame_0.system
#print(traj.get_frames(0,1002))
#pw.MolecularSystem.decipher_atom_keys('self')
frame_0.decipher_atom_keys()
some_forcefield = {
                      'C1': 'C',
                      'C2': 'N',
                      'C3': 'H',
                      'C4': 'H',
                      'C4': 'H',
                      'C4': 'H',
                      'C4': 'H',
                      'C4': 'H',
                      'C4': 'H',

                      }
#traj.analysis()
traj.save_analysis("HISTORY_out.json")
traj.analysis_output

with open("HISTORY_out.json", 'r') as file:
    saved_analysis = json.load(file) 

windows= []
pore_diam_opt = []
max_diam = []

#for key in saved_analysis:
#    print(key)
#    if int(key) >= 200:
#        for i in saved_analysis[key]['0']['windows']['diameters']:
#            windows.append(i)
#        pore_diam_opt.append(saved_analysis[key]['0']['pore_diameter_opt']['diameter'])
#        max_diam.append(saved_analysis[key]['0']['maximum_diameter']['diameter'])
#x_range_windows = np.linspace(min(windows)-1, max(windows)+1, 1000)

#kde_windows = stats.gaussian_kde(windows)
#dist_windows = kde_windows(x_range_windows)

#x_range_pore = np.linspace(min(pore_diam_opt)-1, max(pore_diam_opt)+1, 1000)

#kde_pore = stats.gaussian_kde(pore_diam_opt)
#dist_pore = kde_pore(x_range_pore)

#x_range_max = np.linspace(min(max_diam)-1, max(max_diam)+1, 1000)

#kde_max = stats.gaussian_kde(max_diam)
#dist_max = kde_max(x_range_max)
#fig, ax = plt.subplots(figsize=(7,2.5))

#plt.plot(x_range_windows, dist_windows, label="windows diameter", linewidth=2)

#ax.axes.get_yaxis().set_visible(False)
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(True)
#ax.spines['bottom'].set_color('k')
#ax.spines['left'].set_visible(False)
#ax.grid(b=False)
#ax.set_facecolor('white')
#ax.tick_params(axis='both', which='major', labelsize=12, top='off')
#[t.set_color('k') for t in ax.xaxis.get_ticklabels()]
#[text.set_color("k") for text in ax.legend(frameon=False, fontsize=10, loc=2).get_texts()]
#ax.set_xlabel("Diameter ($\mathregular{\AA)}$", fontsize=12)

plt.tight_layout()
#plt.savefig("trajectory_windows.pdf", dpi=300)
plt.show()
