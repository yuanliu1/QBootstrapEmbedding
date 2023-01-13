import json
import numpy as np



def post_process_swap_counts(swap_result, f0_target, f1_target):
    '''
    input: simulation results from simulator for the swap test
           measurement results corresponds to the evaluation of frag 0
           measurement results corresponds to the evaluation of frag 1
    output: overlap Tr[\rho_A \rho_B]
    '''
    num_evaluation_qubits = int((len(swap_result[0])-1) / 2)
    total_counts = 0.0
    hit_counts = 0.0
    hit_counts_m1 = 0.0
    #print(num_evaluation_qubits)

    for count in swap_result:
        total_counts += 1
        # print(count)
        # print(count[num_evaluation_qubits:-2])
        if count[0:num_evaluation_qubits] == f1_target and count[num_evaluation_qubits:-1] == f0_target:
            hit_counts += 1
            hit_counts_m1 += float(count[-1])
    #print('(qpe post-selection) hit_counts', hit_counts)
    #print('SWAP measurement counts', hit_counts_m1)
    #print(total_counts)

    overlap = hit_counts_m1 / hit_counts
    # print('Prob[M = 1] = %.4e' % overlap) 
    overlap = 1 - 2*overlap
    #print('Overlap Tr[rho_A rho_B] = %.4e' % overlap) 

    return overlap


f0_target_energy = '11011'
f1_target_energy = '11011'

memory = []
ovlp_size = 4
inputname = "h4_sim_ovlp" + str(ovlp_size) + "_2022-11-*.json"
outputname = 'h4_sim_ovlp' + str(ovlp_size) + '_reblocking_not-squared.txt'

## First read in the simulated results
import glob, os
#os.chdir("")
for filename in glob.glob(inputname):
    print(filename)
    with open(filename, "r") as f:
        data = json.load(f)
        memory += data

# with open("h4_sim_ovlp1_2022-11-13-13-15-40.json", "r") as f:
#     data = json.load(f)
#     memory += data
# with open("h4_sim_ovlp1_2022-11-13-13-16-23.json", "r") as f:
#     data = json.load(f)
#     memory += data
# with open("h4_sim_ovlp1_2022-11-13-14-05-14.json", "r") as f:
#     data = json.load(f)
#     memory += data
# with open("h4_sim_ovlp1_2022-11-13-14-05-28.json", "r") as f:
#     data = json.load(f)
#     memory += data
# with open("h4_sim_ovlp1_2022-11-13-14-05-45.json", "r") as f:
#     data = json.load(f)
#     memory += data
# with open("h4_sim_ovlp1_2022-11-13-14-06-36.json", "r") as f:
#     data = json.load(f)
#     memory += data
# with open("h4_sim_ovlp1_2022-11-13-14-06-43.json", "r") as f:
#     data = json.load(f)
#     memory += data
# with open("h4_sim_ovlp1_2022-11-13-14-07-32.json", "r") as f:
#     data = json.load(f)
#     memory += data
#print(score)

print(len(memory))



## Now reblocking to post process the results
f_block_name = outputname 
f_block = open(f_block_name, "w")

blocksize = 40
nblocks = [10,100,1000,10000,1e5]
#nblocks = [4e6]
for numb in nblocks:
    iovlp = []
    for ib in range(int(numb)):
      start_index = ib*blocksize
      end_index = start_index + blocksize
      # now post-selection on the memory
      ovlp2 = post_process_swap_counts(memory[start_index:end_index], f0_target_energy, f1_target_energy)
      iovlp.append(ovlp2)
      #if ovlp2 > 0:
      #    iovlp.append(np.sqrt(ovlp2))
    
    print("%d out of %d blocks have positive value on ovlp2." % (len(iovlp), numb))
    iovlp_avg = np.mean(iovlp)
    iovlp_avg = np.sqrt(iovlp_avg)
    iovlp_err = np.std(iovlp)
    iovlp_err = iovlp_err/(2*iovlp_avg)
    #f_block.write("%8d\t%12.4e\t%12.4e\n" % (len(iovlp) * blocksize, iovlp_avg, iovlp_err/np.sqrt(len(iovlp)-1)))
    f_block.write("%8d\t%12.4e\t%12.4e\n" % (len(iovlp) * blocksize, iovlp_avg, iovlp_err/np.sqrt(len(iovlp)-1)))

f_block.close()
