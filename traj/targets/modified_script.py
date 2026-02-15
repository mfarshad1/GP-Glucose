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

run_1 = 1
pp = PdfPages('hist+pmf'+str(run_1)+'_{real_name}.pdf')
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(hspace=0.0)

# Placeholder for real_name
fname1 = glob.glob('/afs/crc/group/whitmer/Data-MF-'+str(run_1).zfill(2)+'/amber/{real_name}/Umbrella_sampling/umbrella/run_long_'+str(run_1)+'/histo.xvg')   

x = [np.loadtxt(f, usecols=(0,), skiprows=18) for f in fname1]

for i in range (0,{length}):
    y = [np.loadtxt(f, usecols=(i+1,), skiprows=18) for f in fname1]
    y = masked_y = np.ma.masked_where(y[0] == 0, y[0])
    axes[0].plot(x[0], y, '-')

axes[0].set_xticks(np.arange(0, max(x[0])+0.2, 0.2))
axes[0].set_ylabel(r'$\textbf{Counts}$', labelpad=20, fontsize=28)
axes[1].set_xlim(0.0,2.5)
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,3))
axes[0].set_xticklabels([])

fname1 = glob.glob('/afs/crc/group/whitmer/Data-MF-'+str(run_1).zfill(2)+'/amber/{real_name}/Umbrella_sampling/umbrella/run_long_'+str(run_1)+'/profile.xvg') 
y = [np.loadtxt(f, usecols=(1,), skiprows=18) for f in fname1]
x = [np.loadtxt(f, usecols=(0,), skiprows=18) for f in fname1]

# Determine the minimum value
y_min = np.min(y[0])
# Shift the y-values
y = y - y_min

axes[1].plot(x[0], y[0], '-', c = 'b',label='US')
# axes.fill_between(X_, Y_ - statistics.stdev(Y_), Y_ + statistics.stdev(Y_), alpha=0.35)
axes[1].fill_between(x[0], y[0] - statistics.stdev(y[0]), y[0] + statistics.stdev(y[0]), alpha=0.35)

axes[1].set_xticks(np.arange(0, max(x[0])+0.2, 0.2))
axes[1].set_xlabel(r'$\mathbf{\Delta r\ \textbf{(\textit{nm})}}$',fontsize=28)

# axes[1].set(ylabel=r'\boldmath{$\Delta E^{*2}$}')
axes[1].set_ylabel(r'$\textbf{PMF (\textit{kcal/mol})}$', labelpad=16, fontsize=28)
# axes[1].set(ylabel=r'\boldmath{$\Delta E^{*2}/T^{*2}$}')

y = y[0]
x = x[0]


# Apply a smoothing filter to the data
y_smoothed = savgol_filter(y, 31,3)

# Find the global minima and maxima
# Assuming y_smoothed is a NumPy array
min_idx = None
for i in range(1, len(y_smoothed) - 1):
    if x[i] < 0.6:    
        if y_smoothed[i] < y_smoothed[i-1] and y_smoothed[i] < y_smoothed[i+1]:
            min_idx = i
    if x[i] < 1.5:
        if y_smoothed[i] > y_smoothed[i-1] and y_smoothed[i] > y_smoothed[i+1]:
            max_idx = i
# min_idx = np.argmin(y_smoothed)
# max_idx = np.argmax(y_smoothed[min_idx:])+min_idx
diff = y_smoothed[max_idx] - y_smoothed[min_idx]
text_x = np.max(x[min_idx:max_idx])*1.2
text_y = ((y_smoothed[min_idx] + y_smoothed[max_idx])/2)*-0.65
min_x = x[min_idx]
max_x = x[max_idx]

# Plot the data and the absolute minimum and subsequent most smoothed maximum
axes[1].plot(x, y_smoothed, 'r',label='Smoothed line')
axes[1].plot(x[min_idx], y_smoothed[min_idx], 'co', markersize=18)
axes[1].plot(x[max_idx], y_smoothed[max_idx], 'mo', markersize=18)
axes[1].text(text_x, text_y, '$\Delta G^\ddagger = {:.2f}\ kcal/mol$'.format(diff), fontsize=32, ha='center', va='bottom')
# Draw horizontal and vertical lines to connect the minimum and maximum points
axes[1].plot([min(x), max(x)], [y_smoothed[min_idx], y_smoothed[min_idx]], 'k--')
axes[1].plot([max_x, max_x], [y_smoothed[min_idx], y_smoothed[max_idx]], 'k--')
axes[1].legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='22', loc=1, handlelength=0.5)
# axes[0].set_yscale('log')
# axes[1].set_ylim(-6,3.5)
axes[1].set_xlim(0.0,2.5)
axes[1].xaxis.set_ticks(np.arange(0.4, 2.7, 0.4))

# axes.ticklabel_format(style='sci', axis='y', scilimits=(0,3)

# Reverse the y-values
y = y[max_idx] - y

# axes[2].plot(x, y, '-', c = 'b',label='US')
# axes[2].set_xticks(np.arange(0.4, max(x)+0.2, 0.4))
# axes[2].set_xlabel(r'$\mathbf{\Delta r\ \textbf{(\textit{nm})}}$',fontsize=28)
# # axes[1].set(ylabel=r'\boldmath{$\Delta E^{*2}$}')
# axes[2].set_ylabel(r'$\textbf{Reverse PMF (\textit{kcal/mol})}$',fontsize=28)
# # axes[1].set(ylabel=r'\boldmath{$\Delta E^{*2}/T^{*2}$}')
first_positive_index = np.argmax(y > 0)

# Constants
R = 1.98720425864083 # Gas constant
T = 300    # Temperature in Kelvin
conversion = 4.1839953808691
# Calculate the sum of the exponential function
bound = np.sum(np.exp(y[0:max_idx]/(R*T/1000)))
unbound = np.sum(np.exp(y[max_idx+1:]/(R*T/1000)))
constant = bound/unbound
standard_constant = constant
print(standard_constant)
free_energy_cal = -(R*T*np.log(standard_constant))/1000
free_energy_J = conversion*free_energy_cal

#with correction
# Constants
R = 1.98720425864083 # Gas constant
T = 300    # Temperature in Kelvin
conversion = 4.1839953808691
# Calculate the sum of the exponential function
bound = np.sum(np.exp(y[0:max_idx]/(R*T/1000))*((1)/(4/3*np.pi*x[0:max_idx]**3)))
unbound = np.sum(np.exp(y[max_idx+1:]/(R*T/1000))*((1.661)/(4/3*np.pi*x[max_idx+1:]**3)**2))
constant = bound/unbound
standard_constant = constant
standard_constant = constant
print(standard_constant)
free_energy_cal = -(R*T*np.log(standard_constant))/1000
free_energy_J = conversion*free_energy_cal

R = 1.98720425864083 # Gas constant
T = 300    # Temperature in Kelvin
conversion = 4.1839953808691
# Calculate the sum of the exponential function
bound = np.sum(np.exp(y[0:max_idx]/(R*T/1000)))
unbound = np.sum(np.exp(y[max_idx+1:]/(R*T/1000)))
constant = bound/unbound
standard_constant = constant*(max(x)**3)/(1.661)
print(standard_constant)
free_energy_cal = -(R*T*np.log(standard_constant))/1000
free_energy_J = conversion*free_energy_cal

# axes[2].plot(x, np.exp(y/(R*T/1000)), '-', c = 'k',label='US')
# axes[2].set_xticks(np.arange(0.4, max(x)+0.2, 0.4))
# axes[2].set_xlabel(r'$\mathbf{\Delta r\ \textbf{(\textit{nm})}}$',fontsize=28)
# # axes[1].set(ylabel=r'\boldmath{$\Delta E^{*2}$}')
# axes[2].set_ylabel(r'$\textbf{Exp. Factor}$', labelpad=6, fontsize=28)
# # axes[1].set(ylabel=r'\boldmath{$\Delta E^{*2}/T^{*2}$}')
# axes[2].fill_between(x, np.exp(y/(R*T/1000)), where=(x <= x[max_idx]), color='r')
# axes[2].fill_between(x, np.exp(y/(R*T/1000)), where=(x > x[max_idx]), color='b')
# axes[2].set_xlim(0.4,3)
# axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,3))

print("Delta G =", free_energy_cal,' kcal/mol')
print("Delta G =", free_energy_J,' kJ/mol')

# plt.tight_layout()
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()

