import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

fontProperties = {'family':'helvetica','weight':'normal','style':'normal', 'size':20}
matplotlib.rc('font', **fontProperties)

csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}

rmse = np.load('data/rmse_penalty_H8.npy')
rmse = rmse[:11]
rmse2 = np.load('data/rmse_linear_H8_002.npy')
rmse2 = rmse2[:11]
print(rmse)

#energy_error = np.load('energy_error_penalty_H8.npy')
#energy_error = np.load('energy_quad_direct_trace.npy')
#energy_error = np.load('energy_quad_proj_genv_overcount.npy')
#energy_error = np.load('energy_quad_proj.npy')
energy_error = np.load('data/energy_quad_proj_12222022.npy')
energy_error_last = energy_error[11]
energy_error = energy_error[:11]
energy_error = abs(energy_error[:11] - energy_error_last)


#energy_error2 = np.load('energy_error_H8_002.npy')
#energy_error2 = np.load('energy_lin_direct_trace.npy')
#energy_error2 = np.load('energy_lin_proj_genv_overcount.npy')
#energy_error2 = np.load('energy_lin_proj.npy')
energy_error2 = np.load('data/energy_lin_proj_12222022.npy')
energy_error2_last = energy_error2[11]
energy_error2 = energy_error2[:11]
energy_error2 = abs(energy_error2[:11] - energy_error2_last)

fig = plt.figure(figsize=(9,9), dpi = 300)
gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1,1]) 

ax2 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax2)

x = np.arange(0,len(rmse),1)
x2= np.arange(0,len(rmse2),1)

print(x)
print(x2)
print(rmse)
print(rmse2)

#ax2.scatter([0,12],[rmse2[0],1e-5], 'cornflowerblue', label = 'CBE: VMC (?)')
ax2.scatter(x2, rmse2, marker = 'd', color = 'lightcoral', label = 'QBE (Linear)',s=60)

x2_fitted = np.linspace(0, 10, 100)
ax2.plot(x2_fitted, np.exp(np.poly1d(np.polyfit(x2, np.log(rmse2), 1))(x2_fitted)), linestyle='dashed', color='lightcoral',linewidth=1)

# ----- uncomment when quadratic results are available ---
ax2.scatter(x, rmse, marker = 'o', color = 'firebrick', label = 'QBE (Quadratic)', s=60)
x_fitted = np.linspace(0, 10, 100)
ax2.plot(x_fitted, np.exp(np.poly1d(np.polyfit(x, np.log(rmse), 1))(x_fitted)), linestyle='dashed', color='firebrick',linewidth=1)
# ----- uncomment when quadratic results are available ---

# ax2.grid()
ax2.set_yscale("log")
ax2.legend(fontsize=20,loc="upper right")
ax2.set_ylabel('Density Mismatch', **hfont, fontsize=20)
#plt.setp(ax2.get_xticklabels(), visible=True)
#ax2.yticks(fontsize=20, **hfont)
#ax2.tick_params(axis='both', which='major', labelsize=20, **hfont)


absolute_energy = 1
#absolute_energy = 2**14
###### Linear constraint energy error ###################
ax1.scatter(x2, energy_error2 / absolute_energy, marker = 'd', color = 'lightcoral', s=60)
#ax1.plot(x2, energy_error2 / absolute_energy, '--', color = 'lightcoral')
x2_fitted = np.linspace(0, 10, 100)
ax1.plot(x2_fitted, np.exp(np.poly1d(np.polyfit(x2, np.log(energy_error2), 1))(x2_fitted)), linestyle='dashed', color='lightcoral',linewidth=1)

# ----- uncomment when quadratic results are available ---
##### Quadratic constraint energy error ##################
ax1.scatter(x, energy_error / absolute_energy, marker = 'o', color = 'firebrick', s=60)
#ax1.plot(x, energy_error / absolute_energy, '--', color = 'firebrick')
x_fitted = np.linspace(0, 10, 100)
ax1.plot(x_fitted, np.exp(np.poly1d(np.polyfit(x, np.log(energy_error), 1))(x_fitted)), linestyle='dashed', color='firebrick',linewidth=1)
#ax1.plot(x_fitted, np.polyfit(x, energy_error, 2)(x_fitted), linestyle='dashed', color='firebrick',linewidth=1)
# ----- uncomment when quadratic results are available ---

ax1.set_xlabel('Iteration Number', **hfont, fontsize=20)
ax1.set_ylabel('Energy Error', **hfont, fontsize=20)
ax1.set_yscale("log")
plt.xticks([0,1,2,3,4,5,6,7,8,9,10], fontsize=20, **hfont)



locmaj = matplotlib.ticker.LogLocator(base=10,numticks=10)
locmaj_s = matplotlib.ticker.LogLocator(base=100,numticks=10)
#ax1.yaxis.set_minor_locator(AutoMinorLocator(10))
#ax2.yaxis.set_minor_locator(AutoMinorLocator(10))

ax1.yaxis.set_major_locator(locmaj)
ax2.yaxis.set_major_locator(locmaj)

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
locmin_s = matplotlib.ticker.LogLocator(base=100.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
ax1.yaxis.set_minor_locator(locmin)
ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.yaxis.set_minor_locator(locmin)
ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


#plt.yticks(fontsize=20, **hfont)
ax2.yaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)

#plt.show()
# ax1.grid()
plt.savefig('fig4_h8_lin-quad-convergence.png', bbox_inches = 'tight', dpi = 300)






#------------------------------------------
# Fit the function a * np.exp(b * t) - a to x and y
#popt2, pcov2 = curve_fit(lambda t2, a2, b2, c2: a2 * np.exp(b2 * t2) - c2, x2, rmse2, maxfev = 2000)
#a2 = popt2[0]
#b2 = popt2[1]
#c2 = popt2[2]
#print(popt2)
#x2_fitted = np.linspace(1, 10, 100)
#rmse2_fitted = a2 * np.exp(b2 * x2_fitted) - c2
#ax2.plot(x2_fitted, rmse2_fitted, linestyle='dashed', color='cornflowerblue',linewidth=1)
#------------------------------------------



