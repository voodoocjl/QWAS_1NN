import pickle
import os
from Arguments import Arguments
import random
import json
import csv
import numpy as np
import torch
from Node import Node, Color
# from schemes import Scheme
from FusionModel import cir_to_matrix 
import time
from sampling import sampling_node
import copy
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from prepare import *
from draw import plot_2d_array

class MCTS:
    def __init__(self, search_space, tree_height, arch_code):
        assert type(search_space)    == type([])
        assert len(search_space)     >= 1
        assert type(search_space[0]) == type([])

        self.search_space   = search_space 
        self.ARCH_CODE      = arch_code
        self.ROOT           = None
        self.Cp             = 0.2
        self.nodes          = []
        self.samples        = {}
        self.samples_true   = {}
        self.samples_compact = {}
        self.TASK_QUEUE     = []
        self.DISPATCHED_JOB = {}
        self.mae_list    = []
        self.JOB_COUNTER    = 0
        self.TOTAL_SEND     = 0
        self.TOTAL_RECV     = 0
        self.ITERATION      = 0
        self.MAX_MAEINV     = 0
        self.MAX_SAMPNUM    = 0
        self.sample_nodes   = []
        self.stages         = 0
        self.sampling_num   = 0   
        self.acc_mean       = 0

        self.tree_height    = tree_height

        # initialize a full tree
        total_nodes = 2**tree_height - 1
        for i in range(1, total_nodes + 1):
            is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 0:
                is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 1:
                is_good_kid = True

            parent_id = i // 2 - 1
            if parent_id == -1:
                self.nodes.append(Node(None, is_good_kid, self.ARCH_CODE, True))
            else:
                self.nodes.append(Node(self.nodes[parent_id], is_good_kid, self.ARCH_CODE, False))

        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT
        self.weight = 'init'        
        self.explorations = {'phase': 0, 'iteration': 0, 'single':None, 'enta': None, 'rate': [0.0005, 0.001, 0.002], 'rate_decay': [0.006, 0.004, 0.002, 0]}
        self.best = {'acc': 0, 'model':[]}
        self.task = ''
        self.history = [[], []]
        self.qubit_used = []

    def set_init_arch(self, arch):
        single = arch[0]
        enta =arch[1]
        self.explorations['single'] = single
        self.explorations['enta'] = enta
            
    
    def init_train(self, numbers=50):
        # random.seed(40)
        for i in range(0, numbers):
            net = random.choice(self.search_space)
            self.search_space.remove(net)
            self.TASK_QUEUE.append(net)
            self.sample_nodes.append('random')        
        
        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for initializing MCTS")

    def re_init_tree(self, mode=None):
        
        self.TASK_QUEUE = []
        self.sample_nodes = []
        
        self.stages += 1
        phase = self.explorations['phase']        
        # strategy = 'base'
        strategy = self.weight
        round = 3

        if self.task != 'MOSI':
            sorted_changes = [k for k, v in sorted(self.samples_compact.items(), key=lambda x: x[1], reverse=True)]
            epochs = 20
            samples = 10
            file_single = 'search_space_glass_single'
            file_enta = 'search_space_glass_enta'
        else:
            sorted_changes = [k for k, v in sorted(self.samples_compact.items(), key=lambda x: x[1])]
            epochs = 3
            samples = 30
            file_single = 'search_space_mosi_single'
            file_enta = 'search_space_mosi_enta'
        sorted_changes = [change for change in sorted_changes if len(eval(change)) == self.stages]

        # pick best 2 and randomly choose one
        random.seed(self.ITERATION)
        
        best_changes = [eval(sorted_changes[i]) for i in range(2)]
        best_change = random.choice(best_changes)
        self.ROOT.base_code = best_change
        qubits = [code[0] for code in self.ROOT.base_code]
        if self.stages != round and self.history[phase] != []:
            qubits += self.history[phase][-1]
        print('Current Change: ', best_change)
        
        # with open('data/best_changes', 'wb') as file:
        #     pickle.dump(best_change, file)

        if phase == 0:
            best_change_full = self.insert_job(self.explorations['single'], best_change)
            single = best_change_full
            enta = self.explorations['enta']
        else:
            best_change_full = self.insert_job(self.explorations['enta'], best_change)
            single = self.explorations['single']
            enta = best_change_full
        arch = cir_to_matrix(single, enta, self.ARCH_CODE)
        # plot_2d_array(arch)
        design = translator(single, enta, 'full', self.ARCH_CODE)        
        best_model, report = Scheme(design, self.task, strategy, epochs)
        self.weight = best_model.state_dict()
        self.samples_compact = {}
        if report['mae'] > self.best['acc']:
            self.best['acc'] = report['mae']
            self.best['model'] = arch

        import datetime        
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%m-%d-%H')
        torch.save(best_model.state_dict(), 'weights/weight_{}_{}'.format(self.ITERATION, formatted_time))

        with open('results_{}_fine.csv'.format(self.task), 'a+', newline='') as res:
            writer = csv.writer(res)
            metrics = report['mae']
            writer.writerow([self.ITERATION, best_change_full, metrics])
        
        if self.stages == 3:
            self.stages = 0
            self.history[phase].append(qubits)
            phase = 1 - phase       # switch phase
            self.ROOT.base_code = None
            if self.history[phase] != []:
                qubits = self.history[phase][-1]
            else:
                qubits = []
            self.qubit_used = qubits
            self.set_arch(phase, best_change_full)
            self.samples_compact = {}
            self.explorations['iteration'] += 1
            print(Color.BLUE + 'Phase Switch: {}'.format(phase) + Color.RESET)

        
        if phase == 0:
            filename = file_single
        else:
            filename = file_enta
        with open(filename, 'rb') as file:
            search_space = pickle.load(file)
        self.search_space = [x for x in search_space if x[0] not in qubits] 

        random.seed(self.ITERATION)

        for i in range(0, samples):
            net = random.choice(self.search_space)
            while net[0] in qubits:
                net = random.choice(self.search_space)
            self.search_space.remove(net)
            if self.ROOT.base_code != None:
                net_ = self.ROOT.base_code.copy()
                net_.append(net)
            else:
                net_ = [net]
            self.TASK_QUEUE.append(net_)
            self.sample_nodes.append('random')
        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for re-initializing MCTS {}".format(self.ROOT.base_code))

        self.qubit_used = qubits

        # if self.stages == 3:
        #     # self.init_train()
        #     self.stages = 0
        # else:
        #     for i in range(0, samples):
        #         net = random.choice(self.search_space)
        #         while net[0] in qubits:
        #             net = random.choice(self.search_space)
        #         self.search_space.remove(net)
        #         net_ = self.ROOT.base_code.copy()
        #         net_.append(net)
        #         self.TASK_QUEUE.append(net_)
        #         self.sample_nodes.append('random')

            # print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for re-initializing MCTS {}".format(self.ROOT.base_code))    

    def dump_all_states(self, num_samples):
        node_path = 'states/mcts_agent'
        with open(node_path+'_'+str(num_samples), 'wb') as outfile:
            pickle.dump(self, outfile)


    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()

    def set_arch(self, phase, best_change):
        if phase == 0:            
            self.explorations['enta'] = best_change
            # self.explorations['single'] = None
        else:
            self.explorations['single'] = best_change
            # self.explorations['enta'] = None

        self.explorations['phase'] = phase        


    def populate_training_data(self):
        self.reset_node_data()
        for k, v in self.samples.items():
            self.ROOT.put_in_bag(json.loads(k), v)    


    def populate_prediction_data(self):
        # self.reset_node_data()
        for k in self.search_space:
            self.ROOT.put_in_bag(k, 0.0)


    def train_nodes(self):
        for i in self.nodes:
            i.train()


    def predict_nodes(self, method = None, dataset =None):
        for i in self.nodes:            
            if dataset:
                i.predict_validation()
            else:
                i.predict(self.explorations, method)


    def check_leaf_bags(self):
        counter = 0
        for i in self.nodes:
            if i.is_leaf is True:
                counter += len(i.bag)
        assert counter == len(self.search_space)


    def reset_to_root(self):
        self.CURT = self.ROOT


    def print_tree(self):
        print('\n'+'-'*100)
        for i in self.nodes:
            print(i)
        print('-'*100)


    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        self.ROOT.counter += 1
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(self.Cp))
            curt_node = curt_node.kids[np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]]
            self.nodes[curt_node.id].counter += 1
        return curt_node

    def insert_job(self, change_code, job_input):
        job = copy.deepcopy(job_input)
        if type(job[0]) == type([]):
            qubit = [sub[0] for sub in job]
        else:
            qubit = [job[0]]
            job = [job]
        if change_code != None:            
            for change in change_code:
                if change[0] not in qubit:
                    job.append(change)
        return job


    def evaluate_jobs_before(self):
        jobs = []
        designs =[]        
        archs = []
        nodes = []
        while len(self.TASK_QUEUE) > 0:            
           
            job = self.TASK_QUEUE.pop()
            sample_node = self.sample_nodes.pop()               
            if type(job[0]) != type([]):
                job = [job]                             
            if self.explorations['phase'] == 0:
                single = self.insert_job(self.explorations['single'], job)
                enta = self.explorations['enta']
            else:
                single = self.explorations['single']
                enta = self.insert_job(self.explorations['enta'], job)
            design = translator(single, enta, 'full', self.ARCH_CODE)
            arch = cir_to_matrix(single, enta, self.ARCH_CODE)
            
            jobs.append(job)
            designs.append(design)
            archs.append(arch)
            nodes.append(sample_node)

        return jobs, designs, archs, nodes

    def evaluate_jobs_after(self, results, jobs, archs, nodes):
        for i in range(len(jobs)):
            acc = results[i]
            job = jobs[i]                    
            job_str = json.dumps(job)
            arch = archs[i]
            arch_str = json.dumps(np.int8(arch).tolist())
            
            self.DISPATCHED_JOB[job_str] = acc
            if self.task != 'MOSI':
                exploration, gate_numbers = count_gates(arch, self.explorations['rate'])
            else:
                if self.explorations['phase'] == 0:
                    zero_counts = [job[i].count(0) for i in range(len(job))]
                    gate_reduced = np.sum(zero_counts)
                else:
                    zero_counts = [(job[i].count(job[i][0])-1) for i in range(len(job))]
                    gate_reduced = np.sum(zero_counts)
                exploration = gate_reduced * self.explorations['rate_decay'][self.stages]
            p_acc = acc - exploration
            # p_acc = acc
            self.samples[arch_str] = p_acc
            self.samples_true[arch_str] = acc
            self.samples_compact[job_str] = p_acc
            sample_node = nodes[i]
            with open('results_{}.csv'.format(self.task), 'a+', newline='') as res:
                writer = csv.writer(res)                                        
                num_id = len(self.samples)
                writer.writerow([num_id, job_str, sample_node, acc, p_acc])
            self.mae_list.append(acc)
                        
            # if job_str in dataset and self.explorations['iteration'] == 0:
            #     report = {'mae': dataset.get(job_str)}
            #     # print(report)
            # self.explorations[job_str]   = ((abs(np.subtract(self.topology[job[0]], job))) % 2.4).round().sum()
            

    def early_search(self, iter):       
        # save current state
        self.ITERATION = iter
        if self.ITERATION > 0:
            self.dump_all_states(self.sampling_num + len(self.samples))
        print("\niteration:", self.ITERATION)
        if self.task == 'MOSI':
            period = 2
            number = 50
        else:
            period = 3
            number = 10

        if (self.ITERATION % period == 0): 
            if self.ITERATION == 0:
                self.init_train(number)                    
            else:
                self.re_init_tree()                                        

        # evaluate jobs:
        print("\nevaluate jobs...")
        self.mae_list = []
        jobs, designs, archs, nodes = self.evaluate_jobs_before()

        return jobs, designs, archs, nodes
    
    def late_search(self, jobs, results, archs, nodes):
                
        self.evaluate_jobs_after(results, jobs, archs, nodes)    
        print("\nfinished all jobs in task queue")            

        # assemble the training data:
        print("\npopulate training data...")
        self.populate_training_data()
        print("finished")

        # training the tree
        print("\ntrain classifiers in nodes...")
        if torch.cuda.is_available():
            print("using cuda device")
        else:
            print("using cpu device")
        
        start = time.time()
        self.train_nodes()
        print("finished")
        end = time.time()
        print("Running time: %s seconds" % (end - start))
       
        # clear the data in nodes
        self.reset_node_data()                      

        print("\npopulate prediction data...")
        self.populate_prediction_data()
        print("finished")        
        print("\npredict and partition nets in search space...")
        self.predict_nodes()        
        self.check_leaf_bags()
        print("finished")
        self.print_tree()
        # # sampling nodes
        # # nodes = [0, 1, 2, 3, 8, 12, 13, 14, 15]
        # nodes = [0, 3, 12, 15]
        # sampling_node(self, nodes, dataset, self.ITERATION)
        
        random.seed(self.ITERATION)
        print('Used Qubits:', self.qubit_used)
        for i in range(0, 10):
            # select
            target_bin   = self.select()
            # if self.ROOT.base_code == None:
            #     qubits = None
            # else:
            #     qubits = self.qubit_used
            qubits = self.qubit_used
            sampled_arch = target_bin.sample_arch(qubits)
            # NOTED: the sampled arch can be None 
            if sampled_arch is not None:                    
                # push the arch into task queue
                if json.dumps(sampled_arch) not in self.DISPATCHED_JOB:
                    self.TASK_QUEUE.append(sampled_arch)
                    # self.search_space.remove(sampled_arch)
                    self.sample_nodes.append(target_bin.id-7)
            else:
                # trail 1: pick a network from the left leaf
                for n in self.nodes:
                    if n.is_leaf == True:
                        sampled_arch = n.sample_arch(qubits)
                        if sampled_arch is not None:
                            print("\nselected node" + str(n.id-7) + " in leaf layer")                                
                            # print("sampled arch:", sampled_arch)
                            if json.dumps(sampled_arch) not in self.DISPATCHED_JOB:
                                self.TASK_QUEUE.append(sampled_arch)
                                # self.search_space.remove(sampled_arch)
                                self.sample_nodes.append(n.id-7)
                                break
                        else:
                            continue
            if type(sampled_arch[0]) == type([]):
                arch = sampled_arch[-1]
            else:
                arch = sampled_arch
            self.search_space.remove(arch)                          


def Scheme_mp(design, job, task, weight, i, q=None):
    step = len(design)    
    if task != 'MOSI':
        from schemes import Scheme
        epoch = 1
    else:
        from schemes_mosi import Scheme
        epoch = 3
    for j in range(step):
        print('Arch:', job[j][-1])
        _, report = Scheme(design[j], task, weight, epoch, verbs=1)
        q.put([i*step+j, report['mae']])

def count_gates(arch, coeff=None):
    # x = [item for i in [2,3,4,1] for item in [1,1,i]]
    qubits = 6
    layers = 4
    x = [[0, 0, i]*4 for i in range(1,qubits+1)]    
    x = np.transpose(x, (1,0))
    x = np.sign(abs(x-arch))
    if coeff != None:
        coeff = np.reshape(coeff * 4, (-1,1))
        y = (x * coeff).sum()
    else:
        y = 0
    stat = {}
    stat['uploading'] = x[[3*i for i in range(layers)]].sum()
    stat['single'] = x[[3*i+1 for i in range(layers)]].sum()
    stat['enta'] = x[[3*i+2 for i in range(layers)]].sum()
    return y, stat

def analysis_result(samples, ranks):
    gate_stat = []    
    sorted_changes = [k for k, v in sorted(samples.items(), key=lambda x: x[1], reverse=True)]
    for i in range(ranks):
        _, gates = count_gates(eval(sorted_changes[i]))
        gate_stat.append(list(gates.values()))
    mean = np.mean(gate_stat, axis=0)
    return mean

def sampling_arch(search_space, qubits):
    arch_list = []
    while len(qubits) > 0:    
        arch = random.sample(search_space, 1)
        if arch[0][0] in qubits:
            qubits.remove(arch[0][0])
            arch_list.append(arch[0])
    return arch_list

def create_agent(task, arch_code, node=None):
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(state_path, x)))
        node_path = os.path.join(state_path, files[-1])
        if node: node_path = node        
        with open(node_path, 'rb') as json_data:
            agent = pickle.load(json_data)
        print("\nresume searching,", agent.ITERATION, "iterations completed before")        
        print("=====>loads:", len(agent.samples), "samples")        
        print("=====>loads:", len(agent.TASK_QUEUE), 'tasks')
    else:
        if task == 'MNIST':
            path = ['search_space_mnist_full_single', 'search_space_mnist_full_enta']
            n_qubit = 10
            n_layer = 4
            n_single = 8
            n_enta = 8
        elif task == 'GLASS':
            path = ['search_space_glass_single', 'search_space_glass_enta']
            n_qubit = 6
            n_layer = 4
            n_single = 6
            n_enta = 6
        else:            
            path = ['search_space_mosi_single', 'search_space_mosi_enta']
            n_qubit = 7
            n_layer = 5
            n_single = 3
            n_enta = 3
        with open(path[0], 'rb') as file:
            search_space_single = pickle.load(file)        

        with open(path[1], 'rb') as file:
            search_space_enta = pickle.load(file)

        agent = MCTS(search_space_single, 4, arch_code)
        agent.task = task       
                  

        empty = empty_arch(n_layer, n_qubit)   #layers, qubits
        qubits = random.sample([i for i in range(1, n_qubit+1)],n_single)
        single = sampling_arch(search_space_single, qubits)

        qubits = random.sample([i for i in range(1, n_qubit+1)],n_enta)
        enta = sampling_arch(search_space_enta, qubits)

        single = agent.insert_job(empty[0], single)
        enta = agent.insert_job(empty[1], enta)

        # strong entanglement
        n_qubits = arch_code[0]
        n_layers = arch_code[1]
        single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
        enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]
        agent.explorations['single'] = single
        agent.explorations['enta'] = enta
        
        design = translator(single, enta, 'full', arch_code)
        if weights and weights[-1] == 'init_weight':            
            agent.weight = torch.load(os.path.join(weight_path, weights[-1]))
        else:
            if task != 'MOSI':
                best_model, report = Scheme(design, task, 'init', 20, None, 'save')
            else:
                best_model, report = Scheme(design, task, 'init', 1)
            agent.weight = best_model.state_dict()
            with open('results_{}_fine.csv'.format(task), 'a+', newline='') as res:
                writer = csv.writer(res)
                writer.writerow([0, [single, enta], report['mae']])            

        agent.set_init_arch([single, enta])
        
    return agent

def insert_job(change_code, job_input):
        job = copy.deepcopy(job_input)
        if type(job[0]) == type([]):
            qubit = [sub[0] for sub in job]
        else:
            qubit = [job[0]]
            job = [job]
        if change_code != None:            
            for change in change_code:
                if change[0] not in qubit:
                    job.append(change)
        return job


if __name__ == '__main__':
    
     # set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    mp.set_start_method('spawn')

    saved = None
    # saved = 'states/mcts_agent_329'
    # task = 'FASHION'
    task = 'GLASS'
    task = 'MNIST'
    if task == 'MNIST':
        from schemes import Scheme
        from FusionModel import translator
        num_processes = 2
        path = ['search_space_mnist_full_single', 'search_space_mnist_full_enta']
    else:
        from schemes_mosi import Scheme
        from Mosi_Model import translator
        num_processes = 1 
    
    check_file(task)
    
    arch_code = [10, 4]  # qubits, layer
    # arch_code = [6, 4]  # qubits, layer
    # arch_code = [7, 5]

    with open(path[0], 'rb') as file:
        search_space_single = pickle.load(file)        

    with open(path[1], 'rb') as file:
        search_space_enta = pickle.load(file)       

    for iter in range(6, 100):
        random.seed(iter)
        n_qubit = arch_code[0]
        n_layer = arch_code[1]
        n_single = n_qubit
        n_enta = n_qubit
        empty = empty_arch(n_layer, n_qubit)   #layers, qubits
        qubits = random.sample([i for i in range(1, n_qubit+1)],n_single)
        single = sampling_arch(search_space_single, qubits)

        qubits = random.sample([i for i in range(1, n_qubit+1)],n_enta)
        enta = sampling_arch(search_space_enta, qubits)

        single = insert_job(empty[0], single)
        enta = insert_job(empty[1], enta)
        
        design = translator(single, enta, 'full', arch_code)        
        if task != 'MOSI':
            best_model, report = Scheme(design, task, 'init', 20, 0)
        else:
            best_model, report = Scheme(design, task, 'init', 1)
        
        with open('results_{}_random.csv'.format(task), 'a+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow([iter, [single, enta], report['mae']])  
        

    
    # plot_2d_array(agent.best['model'])
    # rank = 20
    # print('Gate numbers of top {}: {}'.format(rank, analysis_result(agent.samples_true, rank)))
    
        
