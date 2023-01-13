import numpy,time,itertools
import h5py,numpy

from scipy.optimize import curve_fit

# only use the following two lines on slurm
#import matplotlib
#matplotlib.use('pdf')

import matplotlib.pylab as plt
import matplotlib.ticker 
import sys

#color = iter(plt.cm.rainbow(numpy.linspace(0, 1, count)))
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}

ovlp_exact=numpy.array([2.560418e-01, 5.141330e-01, 2.560418e-01, 1.000000e+00])
#ovlp_exact=numpy.array([2.560418e-01,2.560418e-01,2.560418e-01,2.560418e-01])
ovlp_exact = numpy.sqrt(ovlp_exact)

samp1 = []
samp2 = []
samp3 = []
samp4 = []
ovlp1 = []
ovlp2 = []
ovlp3 = []
ovlp4 = []
err1 = []
err2 = []
err3 = []
err4 = []

with open('data/h4_self/h4_sim_ovlp1_reblocking_not-squared.txt', 'r') as file1:
#with open('h4_self/h4_sim_ovlp1_reblocking.txt', 'r') as file1:
  count = 0
  for line in file1:
    line = line.split('\t')
    samp1.append(float(line[0]))
    ovlp1.append(float(line[1]))
    err1.append(float(line[2]))
    count += 1

with open('data/h4_self/h4_sim_ovlp2_reblocking_not-squared.txt', 'r') as file1:
#with open('h4_self/h4_sim_ovlp2_reblocking.txt', 'r') as file1:
  count = 0
  for line in file1:
    line = line.split('\t')
    samp2.append(float(line[0]))
    ovlp2.append(float(line[1]))
    err2.append(float(line[2]))
    count += 1

with open('data/h4_self/h4_sim_ovlp3_reblocking_not-squared.txt', 'r') as file1:
#with open('h4_self/h4_sim_ovlp3_reblocking.txt', 'r') as file1:
  count = 0
  for line in file1:
    line = line.split('\t')
    samp3.append(float(line[0]))
    ovlp3.append(float(line[1]))
    err3.append(float(line[2]))
    count += 1

with open('data/h4_self/h4_sim_ovlp4_reblocking_not-squared.txt', 'r') as file1:
#with open('h4_self/h4_sim_ovlp4_reblocking.txt', 'r') as file1:
  count = 0
  for line in file1:
    line = line.split('\t')
    samp4.append(float(line[0]))
    ovlp4.append(float(line[1]))
    err4.append(float(line[2]))
    count += 1

x=numpy.arange(2,9,0.01)
x_scatter = numpy.arange(2,9,2)
## ---- compute the required number of samples for accuracy of 1e-3 for SWAP test -----##
y_swap = []
eps = 1e-3
D = 1.0
y_swap.append(numpy.exp(2*numpy.log(err1[3]/eps) + numpy.log(samp1[3])))
y_swap.append(numpy.exp(2*numpy.log(err2[3]/eps) + numpy.log(samp2[3])))
y_swap.append(numpy.exp(2*numpy.log(err3[3]/eps) + numpy.log(samp3[3])))
y_swap.append(1.0/(4*eps))

# -- this is the theoretical complexity
y_swap_theor = (1-ovlp_exact**2) / (8* eps**2)
y_tmg_theor = numpy.exp(x) * D / eps**2

# take the ratio of the two; tmg is from the theoretical curve, swap is from real run
y_ratio = numpy.exp(x) * D * 8.0 / (1-(ovlp_exact[0])**2)
y_ratio_scatter = numpy.exp(x_scatter) * D * 8.0 / (1-ovlp_exact[0]**2)

fig, ax1 = plt.subplots()

## plot the ratio now
ax1.scatter(x_scatter, y_ratio_scatter, marker = 's', color = 'cornflowerblue', facecolor = 'none', s=60)
ax1.plot(x, y_ratio, '--', color = 'cornflowerblue')
#ax1.plot(x, y_swap_theor, '--', color = 'firebrick')

## plot tomography scaling
#ax1.scatter(x, y_tmg_theor, marker = 's', color = 'slategrey', facecolor = 'none', label = 'TMG',s=60)
#ax1.plot(x, y_tmg_theor, '--', color = 'slategrey')

## plot swap test scaling
#ax1.scatter(x, y_swap, marker = 'v', color = 'cornflowerblue', facecolor = 'none', label = 'SWAP',s=60)
#ax1.plot(x, y_swap, '--', color = 'cornflowerblue')
#ax1.plot(x, y_swap_theor, '--', color = 'firebrick')

ax1.set_yscale("log")
ax1.set_xlabel("Number of Overlap Qubits", fontsize=16, **hfont)
ax1.set_ylabel(r"$N_{samp}^{TMG} / N_{samp}^{SWAP}$", fontsize=16, **hfont)
# ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 0.07), fontsize=15)
ax1.tick_params(axis='both', which='major', labelsize=14)
#ax1.set_xticks([2,3,4,5,6,7,8], fontsize=16, **hfont)
#ax1.set_yticks([0,5e8,1e9,1.5e9,2e9,2.5e9,3e9],fontsize=16, **hfont)






### ----- Now add inset to show a representative convergence ----------- #####
left, bottom, width, height = [0.25, 0.56, 0.25, 0.3]
ax = fig.add_axes([left, bottom, width, height])
ax.errorbar(samp1, ovlp1, yerr = err1, marker = 'v', color='firebrick', markerfacecolor = 'none', capsize=5, markersize=6,label="Simulated")
ax.hlines(y=ovlp_exact[0], xmin=3e2, xmax=6e6, linewidth=1.5, color='black', linestyle='--', label="Exact")
#ax.plot(x, numpy.poly1d(numpy.polyfit(size2_fit, t_qpe_fit, 5))(x), linestyle='dashed', color='firebrick',linewidth=1)
#plt.yscale("log")
ax.set_xscale("log")
ax.set_xlim([3e2, 5e6])
ax.set_xlabel("Eigensolver Calls", fontsize=12, **hfont)
ax.set_ylabel("Overlap S", fontsize=12, **hfont)
ax.legend(loc="upper right", fontsize=10)
#ax.set_xticks([1e2,1e3,1e4,1e5,1e6,1e7], fontsize=15, **hfont)
#ax.set_yticks(fontsize=15, **hfont)

## #plt.ylim([1, 10**8])
## plt.yscale("log")

ax.tick_params(axis='both', which='major', labelsize=12)

figname ='./fig6_swap_size.png' 
plt.savefig(figname, bbox_inches = "tight", dpi=300)


