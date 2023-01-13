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


count = 0
size1 = []
size2 = []
t_fci = []
ovlp = []
t_qpe = []

with open('./data/time_fci_davidson.txt', 'r') as file1:
  for line in file1:
    if count > 0 and count < 15:
      line = line.split('\t')
      size1.append(float(line[0]))
      t_fci.append(float(line[1]))
    count += 1

count = 0
with open('./data/time_qpe_max.txt', 'r') as file2:
  for line in file2:
    if count > 0 and count < 14:
      line = line.split('\t')
      size2.append(float(line[0]))
      t_qpe.append(float(line[1]))
      ovlp.append(float(line[3]))
    count += 1


# Reduce the size by omitting some data points
#size1 = size1[2:]
#size2 = size2[2:]
#t_fci = t_fci[2:]
#t_qpe = t_qpe[2:]
#ovlp = ovlp[2:]


t_fci = [ t_fci[i] / t_fci[0] for i in range(len(t_fci)) ]
t_qpe = [ t_qpe[i] / t_qpe[0] / (ovlp[i])**2 for i in range(len(t_qpe)) ]
print(t_fci)
print(t_qpe)
print(ovlp)
#print(size)
#plt.plot(t_fci, label = 'FCI')
#plt.plot(size, t_fci, marker = 'o', label = 'FCI')


size1_fit = size1[8:]
size2_fit = size2[:-3]
t_fci_fit = t_fci[8:]
t_qpe_fit = t_qpe[:-3]

csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}

x=numpy.arange(2,20,0.01)


fig, ax = plt.subplots()
ax.scatter(size2, t_qpe, marker = 'o', label = 'QPE', facecolor='none', color='firebrick',s=60)
ax.plot(x, numpy.poly1d(numpy.polyfit(size2_fit, t_qpe_fit, 5))(x), linestyle='dashed', color='firebrick',linewidth=1)


ax.scatter(size1, t_fci, marker = 's', label = 'FCI', facecolor='none', color='cornflowerblue',s=50)
#plt.semilogy(size1, t_fci, marker = 's', label = 'FCI', markerfacecolor='none', color='cornflowerblue',markersize=8,linestyle='dashed')
#plt.plot(x, numpy.exp(numpy.poly1d(numpy.polyfit(size1_fit, numpy.log(t_fci_fit), 1))(x)), linestyle='dashed', color='cornflowerblue',linewidth=1)

# Fit the function a * np.exp(b * t) - a to x and y
popt, pcov = curve_fit(lambda t, a, b, c: a * numpy.exp(b * t) - c, size1_fit, t_fci_fit)
a = popt[0]
b = popt[1]
c = popt[2]
print(popt)
x_fitted = numpy.linspace(9, 20, 50)
y_fitted = a * numpy.exp(b * x_fitted) - c
ax.plot(x_fitted, y_fitted, linestyle='dashed', color='cornflowerblue',linewidth=1)



plt.xlim([1, 20])
#plt.ylim([1, 10**8])
plt.legend(loc="upper left", fontsize=15)
plt.yscale("log")
plt.xlabel("System Size", fontsize=16, **hfont)
plt.ylabel("Time (normalized)", fontsize=16, **hfont)
plt.xticks([1,3,5,7,9,11,13,15,17,19], fontsize=15, **hfont)
plt.yticks(fontsize=15, **hfont)

locmaj = matplotlib.ticker.LogLocator(base=100,numticks=12)
#ax.yaxis.set_minor_locator(AutoMinorLocator(10))
ax.yaxis.set_major_locator(locmaj)

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


#plt.margins(x = 0.0)
#plt.show()


figname ='./fig3_time_fci_qpe.png' 
plt.savefig(figname, bbox_inches = "tight", dpi=300)


