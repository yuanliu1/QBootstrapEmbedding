import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

#fontProperties = {'family':'helvetica','weight':'normal','style':'normal', 'size':20}
#matplotlib.rc('font', **fontProperties)

csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}


eig_calls = []
eig_calls_fci = []
grad1 = []
grad2 = []
grad3 = []
grad4 = []
grad5 = []
grad6 = []
grad7 = []

# with open('./avg_gradients_comp.txt', 'r') as file1:
#   for line in file1:
#     if count > 0 and count < 101:
#       line = line.split(' ')
#       eig_calls.append(float(line[0]))
#       grad1.append(float(line[1]))
#       grad2.append(float(line[2]))
#       grad3.append(float(line[3]))
#       grad4.append(float(line[4]))
#       grad5.append(float(line[5]))
#       grad6.append(float(line[6]))
#     count += 1

with open('./data/h8_avg_gradients_comp_lr0d05.txt', 'r') as file1:
  count = 0
  for line in file1:
    if count > 0 and count < 101:
      line = line.split(' ')
      eig_calls.append(float(line[0]))
      grad1.append(float(line[1]))
      grad2.append(float(line[2]))
      grad3.append(float(line[3]))
    count += 1


#with open('./dens_error_gd.dat', 'r') as file1:
#with open('./plt_data_h8_fci_step0.5.dat', 'r') as file1:
with open('./data/dens_err_per_fci_callstep0.05.dat', 'r') as file1:
  count = 0
  for line in file1:
    if count >=0 and count < 725:
      if count%6 == 0:
        line = line.split('    ')
        eig_calls_fci.append(float(line[0]))
        grad4.append(3.6*float(line[1]))
    count += 1


# compute the average gradients for the last 20 iterations
start_ind = 50
avg1 = np.mean(grad1[start_ind:])
avg2 = np.mean(grad2[start_ind:])
avg3 = np.mean(grad3[start_ind:])
#avg7 = np.mean(grad7[start_ind])

#conv_limit = np.power(1/np.sqrt(np.power(10,1/lr_exponent)),np.arange(len(plot_data[1,:])))

#plt.semilogy(np.arange(len(grad4)), grad4, '-', color = 'black', label='FCI')
plt.scatter(eig_calls_fci, grad4, marker='o', color = 'black', label='FCI', facecolor='none',s=40)
plt.semilogy(eig_calls_fci, grad4, '--', color = 'black')

#plt.semilogy(plot_data[0,:],(plot_data[1,:]),label='Perturbed H4')
#plt.semilogy(np.arange(len(grad1)), grad1, '-', color = 'darkgrey', label='VMC (40k samples)')
plt.semilogy(eig_calls, grad1, '-', color = 'darkgrey', label='VMC (40k samples)')
plt.axhline(y=avg1, color='darkgrey', linestyle='--')

#plt.semilogy(np.arange(len(grad2)), grad2, '-', color = 'lightskyblue', label='VMC (160k samples)')
plt.semilogy(eig_calls, grad2, '-', color = 'lightskyblue', label='VMC (160k samples)')
plt.axhline(y=avg2, color='lightskyblue', linestyle=':')

#plt.semilogy(np.arange(len(grad3)), grad3, '-', color = 'coral', label='VMC (640k samples)')
plt.semilogy(eig_calls, grad3, '-', color = 'coral', label='VMC (640k samples)')
plt.axhline(y=avg3, color='coral', linestyle='-.')

#plt.semilogy(np.arange(len(grad6)), grad6, '-r', label='960k samples')
#plt.axhline(y=avg6, color='r', linestyle='--')
#plt.semilogy(np.arange(len(grad7)), grad7, label='2048k samples')

#plt.semilogy(np.arange(len(grad480)), grad480,label='# Blocks = 480')
#plt.semilogy(plot_data[0,:],conv_limit**2, label = '${(1/\sqrt{\gamma})}^{2\cdot\mathrm{iter}}$')
#plt.xlabel('number of calls to the eigensolver subroutine')



#plt.xlabel('BE Iterations', fontsize=15, **hfont)
plt.xlabel('Eigensolver Calls', fontsize=15, **hfont)
plt.ylabel('Density Mismatch', fontsize=15, **hfont)
#plt.title('Bootstrap Embedding with VMC solver for perturbed H4')
#plt.grid()
plt.legend(fontsize=15)
plt.xlim(0,800)
plt.ylim(4e-4,4e-3)
#plt.xscale('log')
plt.xticks(fontsize=15, **hfont)
plt.yticks(fontsize=15, **hfont)
#plt.savefig('H4_convergence_rate_VMC_run100_lr0d5_eigsol_constSample.png')
plt.savefig('fig2_fci_vmc_compare.png', bbox_inches='tight', dpi=300)

