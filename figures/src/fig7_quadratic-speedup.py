import matplotlib
import numpy as np
matplotlib.use('pdf')

import matplotlib.pylab as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogFormatter

fontProperties = {'family':'helvetica','weight':'normal','style':'normal', 'size':15}
matplotlib.rc('font', **fontProperties)

# epsilon in terms of iteration number
def epsilon_val(c, epsilon_0, iter):
    return epsilon_0 * c ** iter

# S in terms of iteration number
def S_val(c, epsilon_0, iter):
    return 1 - epsilon_0 - c ** iter

# incoherent method without S dependence
def N_incoherent(c, epsilon_0, iter):
    eps = epsilon_val(c, epsilon_0, iter)
    return 1 / (8 * (eps ** 2))

# AABS method without S dependence
def N_AABS(c, epsilon_0, iter):
    eps = epsilon_val(c, epsilon_0, iter)
    return 2 ** (3/2) / eps * np.log(1 / eps) * np.log2(1/eps)

# incoherent method with S dependence
def N_incoherent_S(c, epsilon_0, iter):
    S = S_val(c, epsilon_0, iter)
    eps = epsilon_val(c, epsilon_0, iter)
    return (1 - S ** 2) / 8 * 1 / (eps ** 2)

# AABS method with S dependence
def N_AABS_S(c, epsilon_0, iter):
    S = S_val(c, epsilon_0, iter)
    eps = epsilon_val(c, epsilon_0, iter)
    a = ((1 + (S ** 2)) / 2) ** 0.5
    n = int(np.ceil(np.log2(1 / eps)))
    delta_k = {}
    for k in range(n):
        delta_k[n-k-1] = np.floor(a * 2**k) / 2**k + 2**(-k-1)
    N_samp = 0
    for i in range(n):
        N_samp += 1 / delta_k[i]
    return N_samp * 1 / (2 * eps) * np.log(1/eps)

# incoherent method based on epsilon
def N_incoherent_eps(eps):
    return 1 / (8 * eps ** 2)

# AABS method based on epsilon
def N_AABS_eps(eps):
    return 2 ** (3/2) / eps * np.log(1 / eps) * np.log2(1/eps)


# incoherent method with S dependence, new version 09/30/2022
def N_incoherent_S_yl(eps, S):
    return (1 - S ** 2) / 8 * 1 / (eps ** 2)

# AABS method with S dependence, new version 09/30/2022
def N_AABS_S_yl(eps, S):
    a = ((1 + (S ** 2)) / 2) ** 0.5
    n = int(np.ceil(np.log2(1 / eps)))
    delta_k = {}
    for k in range(n):
        delta_k[n-k-1] = np.floor(a * 2**k) / 2**k + 2**(-k-1)
    N_samp = 0
    for i in range(n):
        N_samp += 1 / delta_k[i]
    return N_samp * 1 / (2 * eps) * np.log(1/eps)



#### Compute the first epsilon dependent plot data #######
S0=0.4
eps = {}
samp_AABS_eps = {}
samp_incoherent_eps = {}
for i in range(26):
    eps = 10 ** (-(i/3 + 0.05))
    #samp_AABS_eps[eps] = N_AABS_eps(eps)
    #samp_incoherent_eps[eps] = N_incoherent_eps(eps)
    samp_AABS_eps[eps] = N_AABS_S_yl(eps, S0)
    samp_incoherent_eps[eps] = N_incoherent_S_yl(eps, S0)
#### Finish Compute the main plot data #######



#### Compute the second S dependent plot data #######
eps0 = 0.001
S_fitted = np.linspace(0, 1, 10000)
print(S_fitted)
samp_AABS_S = np.zeros(len(S_fitted))
samp_AABS_S_independent = np.zeros(len(S_fitted))
samp_incoherent_S = np.zeros(len(S_fitted))

for i in range(len(S_fitted)):
	samp_AABS_S_independent[i] = N_AABS_eps(eps0)
	samp_AABS_S[i] = N_AABS_S_yl(eps0, S_fitted[i])
	samp_incoherent_S[i] = N_incoherent_S_yl(eps0, S_fitted[i])
#print(S_fitted)
#print(samp_AABS_S)
#print(samp_incoherent_S)
#### Finish Compute the main plot data #######




csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}


####### Plot ################
#fig = plt.figure(figsize=(9,9), dpi = 300)
#gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1,1]) 
#ax1 = plt.subplot(gs[0]) ## Sample versus precision
#ax2 = plt.subplot(gs[1]) ## Sample versus overlap S

fig, ax1 = plt.subplots()


###### -------------------------- Plot Scatter points -------------------########
################ Add VMC data 
# No. of Samples
y = np.array([2000000, 1000000, 750000, 500000, 250000, 100000, 75000, 50000, 25000, 10000, 7500, 5000, 2500])
# sigma/sqrt(N)
x = np.array([0.0002232733900811481, 0.000316459948865502, 0.0003656281256434889, 0.0004482549860980571, 0.0006258376938081179, 0.000990256421920347, 0.00114012643376088, 0.0013964857637061266, 0.00197558969404451, 0.0030957595134518266, 0.003614461121106748, 0.004485947758400014, 0.0063346338835634074])

# sampling over a range of N from the 2 Million energies
xmax = np.array([0.0002232733900811481, 0.00031730020938901493, 0.0003673413524445422, 0.00045072346113009373, 0.0006422115284688396, 0.0010299902798680295, 0.0012003298614108554, 0.0015026564883697778, 0.0022581671711979392, 0.004141686806010231, 0.004595101291117703, 0.007073484082004392, 0.008557663248677336])
xmin = np.array([0.0002232733900811481,0.0003083583719272854, 0.000361764353853275, 0.00044144671236674016, 0.0006249191571159441, 0.0009174450630246503, 0.001136769876241005, 0.0013878665155207615, 0.001941633674142189, 0.003044785087329865, 0.003514620034467456, 0.0042280028329831935, 0.005917825615337971])

xtop = xmax-x
xbot = x-xmin

#fig, ax = plt.subplots()
ax1.errorbar(x,y, xerr=(xbot, xtop),  capsize=5, label='$\mathtt{SWAP}$' ,color='cornflowerblue', marker = '+', markerfacecolor='none', markersize=6)


######### Read in real data from figure 4 folder ########
rmse = np.load('data/rmse_penalty_H8.npy')
rmse2 = np.load('data/rmse_linear_H8_002.npy')


points_AABS = []
for i in range(len(rmse2)):
    eps = rmse2[i]
    points_AABS.append(N_AABS_S_yl(eps,S0))

print('Computed sample numbers using real data:\n', points_AABS)
ax1.scatter(rmse2, points_AABS, marker = 'o', facecolor = 'none', color = 'firebrick', label = '$\mathtt{SWAP}$+AE', s=60)

###### -------------------------- Finish Plot Scatter points -------------------########



###### -------------------------- Plot Dashed Lines -------------------########
ax1.loglog(samp_incoherent_eps.keys(), samp_incoherent_eps.values(), '--', color='cornflowerblue')
ax1.loglog(samp_AABS_eps.keys(), samp_AABS_eps.values(), '--', color='firebrick')
ax1.set_xlabel("Target Precision", fontsize=15, **hfont)
ax1.set_ylabel("Number of Eigensolver Calls", fontsize=15, **hfont)
#ax1.xlabel("Target Precision", **hfont)
#fig.ylabel("Number of Eigensolver Calls", **hfont)

legendfont = font_manager.FontProperties(family='helvetica',weight='normal',style='normal', size=15)
ax1.legend(loc="lower left", prop = legendfont)
ax1.set_xlim(10**(-8), 1.0)
ax1.set_ylim(1.0, 10**16)


#ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
#ax1.set_yticklabels(ax1.get_yticks(), fontProperties)
###### -------------------------- Plot Dashed Lines -------------------########




########### ----------------------------Plot second panel ------------#############
##ax2.semilogy(S_fitted, samp_incoherent_S, '-', color='cornflowerblue')
##ax2.semilogy(S_fitted, samp_AABS_S, '-', color='firebrick')
#ax2.plot(S_fitted, samp_incoherent_S, '-', color='cornflowerblue')
#ax2.plot(S_fitted, samp_AABS_S, '-', color='firebrick')
#ax2.plot(S_fitted, samp_AABS_S_independent, '--', color='firebrick')
#ax2.set_xlabel("Overlap S", fontsize=15, **hfont)
#ax2.set_ylabel("Number of Eigensolver Calls", fontsize=15, **hfont)
#ax2.set_xlim(0, 1.0)
##ax2.set_ylim(40800, 41000)
########### --------------------------Finish Plot second panel ----------#############


###### -------------------------- Plot Inset -------------------########
#iteration_rmse = np.arange(0,len(rmse),1)
#iteration_fitted = np.linspace(0, 150, 1000)
#rmse_fitted = np.exp(np.poly1d(np.polyfit(iteration_rmse, np.log(rmse), 1))(iteration_fitted))
#
##### Compute the Subplot data ######
#samp_incoherent = {}
#samp_AABS = {}
#samp_AABS_S = {}
#samp_incoherent_S = {}
#for i in range(20):
#    iter = 10 * i + 10
#    eps[iter] = epsilon_val(0.93, 0.05, iter)
#    samp_incoherent[iter] = N_incoherent(0.93, 0.05, iter)
#    samp_AABS[iter] = N_AABS(0.93, 0.05, iter)
#    samp_incoherent_S[iter] = N_incoherent_S(0.93, 0.05, iter)
#    samp_AABS_S[iter] = N_AABS_S(0.93, 0.05, iter)
#
#
#
left, bottom, width, height = [0.54, 0.54, 0.33, 0.31]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(S_fitted, samp_incoherent_S, '-', color='cornflowerblue')
ax2.plot(S_fitted, samp_AABS_S, '-', color='firebrick')
#ax2.plot(S_fitted, samp_AABS_S_independent, '--', color='firebrick')
ax2.set_xlabel("Overlap S", fontsize=15, **hfont)
ax2.set_ylabel("",fontsize=15, **hfont)
ax2.set_xlim(0, 1.0)
#ax2.semilogy(samp_incoherent.keys(), samp_incoherent.values(), 'b:', label = 'Incoherent')
#ax2.semilogy(samp_incoherent_S.keys(), samp_incoherent_S.values(), 'b-', label = 'Incoherent (S dependent)')
#ax2.semilogy(samp_AABS.keys(), samp_AABS.values(), ':', color = 'firebrick', label = 'AA+BS')
#ax2.semilogy(samp_AABS_S.keys(), samp_AABS_S.values(), '-', color = 'firebrick', label = 'AA+BS (S dependent)')
##plt.semilogy(eps.keys(), eps.values(), label = 'epsilon')
#ax2.set(xlabel = "Number of iterations")
#
#plt.xticks(fontsize=15,**hfont)
#plt.yticks(fontsize=15,**hfont)
###### -------------------------- Finish Plot Inset -------------------########




locmaj = matplotlib.ticker.LogLocator(base=100,numticks=24)
##ax.yaxis.set_minor_locator(AutoMinorLocator(10))
#ax1.yaxis.set_major_locator(locmaj)
##ax2.yaxis.set_major_locator(locmaj)
ax1.xaxis.set_major_locator(locmaj)
##ax2.xaxis.set_major_locator(locmaj)
#
locmin = matplotlib.ticker.LogLocator(base=100,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=24)
locminy = matplotlib.ticker.LogLocator(base=1000,subs=(0.2,0.4,0.6,0.8),numticks=24)
ax1.yaxis.set_minor_locator(locminy)
ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.xaxis.set_minor_locator(locmin)
ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())



ax2.xaxis.set_major_formatter('{x:1.1f}')
ax2.set_yticks([0,5e4,1e5])
#ax2.yaxis.set_major_formatter('{x:1.0E}')
#ax2.yaxis.set_major_formatter(LogFormatter())
#ax2.ticklabel_format(axis='y', style='sci', scilimits=None, useOffset=None, useLocale=None, useMathText=True)
# For the minor ticks, use no labels; default NullFormatter.
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

#
#
##plt.yticks(fontsize=20, **hfont)
##ax2.yaxis.set_tick_params(labelsize=20)
#ax1.yaxis.set_tick_params(labelsize=20)


#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_xlabel('Epsilon')
#ax.set_ylabel('No. of Samples')
#ax.set_xlim([0.0001, 0.010])
#plt.show()

#fig = plt.gcf()
#fig.set_size_inches(9,6)
#ax1.scatter(x, y, 'b-', label = 'Classical VMC')
#fig.savefig('Fig5_inset.pdf', bbox_inches='tight')





figname = 'fig5-quadratic_speedup.png'
plt.savefig(figname, bbox_inches = 'tight', dpi=300)
