import pickle
import os
import csv

def check_file(task):
    result = 'results_{}.csv'.format(task)
    result_fine = 'results_{}_fine.csv'.format(task)
    if os.path.isfile(result) == False:
        with open(result, 'w+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow(['sample_id', 'arch_code', 'sample_node', 'ACC', 'p_ACC'])

    if os.path.isfile(result_fine) == False:
        with open(result_fine, 'w+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow(['iteration', 'arch_code', 'ACC'])

state_path = 'states'
if os.path.exists(state_path) == False:
    os.makedirs(state_path)
files = os.listdir(state_path)

weight_path = 'weights'
if os.path.exists(weight_path) == False:
    os.makedirs(weight_path)
weights = os.listdir(weight_path)

def empty_arch(n_layers, n_qubits):            
    single = [[i] + [0]* (2*n_layers) for i in range(1,n_qubits+1)]
    enta = [[i] *(n_layers+1) for i in range(1,n_qubits+1)]
    return [single, enta]



