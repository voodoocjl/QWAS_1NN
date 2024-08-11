import json
from math import log2, ceil
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from Network import Attention, RNN, normalize, FC
from FusionModel import cir_to_matrix
import time

torch.cuda.is_available = lambda : False


def get_label(energy, mean = None):
    # label = energy.clone()
    # if mean and (mean < float('inf')):
    #     energy_mean = mean
    # else:
    #     energy_mean = energy.mean()
    # for i in range(energy.shape[0]):
    #     label[i] = energy[i] > energy_mean

    x = energy
    height = 4    #MCTS TREE height
    a = [[i for i in range(len(x))]]
    for i in range(1,height):
        t = []
        for j in range(2**(i-1)):        
            index = a[j]
            if len(index):
                mean = x[index].mean()
            else:
                mean = []
            t.append(torch.tensor([item for item in index if x[item] >= mean]))
            t.append(torch.tensor([item for item in index if x[item] < mean]))
        a = t
    label = torch.zeros((len(x), height-1))
    for i in range(len(a)):
        index = a[i]
        if len(index):
            for j in range(len(index)):
                string_num = bin(i)[2:].zfill(height-1)
                label[index[j]] = torch.tensor([int(char) for char in string_num])
    return label

def insert_job(change_code, job):
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


class Classifier:
    def __init__(self, samples, arch_code, node_id):
        assert type(samples) == type({})        

        self.samples          = samples
        self.arch_code        = arch_code   #[qubits, layers]        
        self.input_dim_2d     = 21
        self.training_counter = 0
        self.node_layer       = ceil(log2(node_id + 2) - 1)
        # self.model            = Linear(self.input_dim_2d, 2)
        # self.model            = Mlp(self.input_dim_2d, 6, 2)        
        # self.model            = RNN(arch_code[0], 16, 2)
        self.model            = FC(arch_code)
        
        self.loss_fn          = nn.CrossEntropyLoss() #nn.MSELoss()
        self.l_rate           = 0.001
        self.optimizer        = optim.Adam(self.model.parameters(), lr=self.l_rate, betas=(0.9, 0.999), eps=1e-08)
        self.epochs           = []
        self.training_accuracy = [0]
        self.boundary         = -1
        self.nets             = None
        self.maeinv           = None
        self.labels           = None
        self.mean             = 0        
        self.period           = 10
        


    def update_samples(self, latest_samples, layer=0, latest_labels=None):
        assert type(latest_samples) == type(self.samples)
        self.samples = latest_samples        
        if layer == 0:
            sampled_nets = []
            nets_maeinv  = []
            for k, v in latest_samples.items():
                net = json.loads(k)            
                sampled_nets.append(net)
                nets_maeinv.append(v)
            self.nets = torch.from_numpy(np.asarray(sampled_nets, dtype=np.float32))
            self.nets = normalize(self.nets)       
            self.maeinv = torch.from_numpy(np.asarray(nets_maeinv, dtype=np.float32).reshape(-1, 1))
            self.labels = get_label(self.maeinv)            
            if torch.cuda.is_available():
                self.nets = self.nets.cuda()
                self.maeinv = self.maeinv.cuda()
                self.labels = self.labels.cuda()
        else:
            self.pred_labels = latest_labels


    def train(self):
        if self.training_counter == 0:
            self.epochs = 1000
        else:
            self.epochs = 500
        self.training_counter += 1
        # in a rare case, one branch has no networks
        if len(self.nets) == 0:
            return
        # linear, mlp
        nets = self.nets
        # labels = 2 * self.labels - 1
        labels = self.labels        
        train_data = TensorDataset(nets, labels)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        if torch.cuda.is_available():
            self.model.cuda()
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in [self.model.cls1.weight, self.model.cls1.bias,
                      self.model.cls2.weight, self.model.cls2.bias,
                      self.model.cls3.weight, self.model.cls3.bias]:
            param.requires_grad = True
                    
        for epoch in range(self.epochs):
            for x, y in train_loader:
                # clear grads
                self.optimizer.zero_grad()
                # forward to get predicted values
                outputs = self.model(x)
                # loss_mae = self.loss_fn(outputs[:, 0], y.reshape(-1))
                loss_t = self.loss_fn(outputs[0], y.long())
                # loss = loss_mae + loss_t
                loss = loss_t
                loss.backward()  # back props
                # grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
                # norm = torch.cat(grads).norm()
                # print('Grad Norm: ', norm)
                # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()  # update the parameters

        # training accuracy
        pred = self.model(nets)
        
        pred_label = pred[1].float().cpu()
        true_label = self.labels.cpu()        
        acc = accuracy_score(true_label.numpy(), pred_label.numpy())
        self.training_accuracy.append(acc)    

    
    def predict(self, remaining, arch):
        assert type(remaining) == type({})
        remaining_archs = []
        for k, v in remaining.items():
            net = json.loads(k)
            if arch['phase'] == 0:
                net = insert_job(arch['single'], net) 
                net = cir_to_matrix(net, arch['enta'], self.arch_code)                
            else:
                net = insert_job(arch['enta'], net)
                net = cir_to_matrix(arch['single'], net, self.arch_code)           
            remaining_archs.append(net)
        remaining_archs = torch.from_numpy(np.asarray(remaining_archs, dtype=np.float32))
        remaining_archs = normalize(remaining_archs)
                
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cuda()
        t1 = time.time()
        outputs = self.model(remaining_archs)
        print('Prediction time: ', time.time()-t1)
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cpu()
            # outputs         = outputs.cpu()
        diff = -(outputs[0][:, 0, :] - outputs[0][:, 1, :]).abs().detach().cpu()
        result = []        
        result.append(list(remaining.keys()))
        result.append( outputs[1].tolist())
        
        assert len(result[0]) == len(remaining)
        return result, diff


    def split_predictions(self, remaining, arch, layer=0, delta = None):
        # assert type(remaining) == type({})
        samples_badness = [[], []]
        samples_goodies = [[], []]
        delta_badness = []
        delta_goodies = []

        if layer == 0:
            predictions, delta = self.predict(remaining, arch)  # arch_str -> pred_test_mae            
        else:
           predictions = remaining
           remaining = remaining[0]
           if len(remaining) == 0:
            return samples_goodies, samples_badness, delta_goodies, delta_badness, []
           

        for index, (k, v) in enumerate(zip(predictions[0], predictions[1])):
            if v[layer] == 1 :                
                samples_badness[0].append(k)
                samples_badness[1].append(v)
                delta_badness.append(index)  # bad index
            else:
                samples_goodies[0].append(k)
                samples_goodies[1].append(v)
                delta_goodies.append(index)
        delta_badness = delta[delta_badness]
        delta_goodies = delta[delta_goodies]
        assert len(samples_badness[0]) + len(samples_goodies[0]) == len(remaining)
        delta = torch.exp(delta).mean(dim=0)
        return samples_goodies, samples_badness, delta_goodies, delta_badness, delta

    """
    def predict_mean(self):
        if len(self.nets) == 0:
            return 0
        # can we use the actual maeinv?
        outputs = self.model(self.nets)
        pred_np = None
        if torch.cuda.is_available():
            pred_np = outputs.detach().cpu().numpy()
        else:
            pred_np = outputs.detach().numpy()
        return np.mean(pred_np)
    """

    def sample_mean(self):
        if len(self.nets) == 0:
            return 0
        outputs = self.maeinv
        true_np = None
        if torch.cuda.is_available():
            true_np = outputs.cpu().numpy()
        else:
            true_np = outputs.numpy()
        return np.mean(true_np)


    def split_data(self, layer=0, f1 = None):
        samples_badness = {}
        samples_goodies = {}
        samples_badness_labels = {}
        samples_goodies_labels = {}        
        if layer == 0:
            if len(self.nets) == 0:
                return samples_goodies, samples_badness
            
            self.train()
            outputs = self.model(self.nets)
            if torch.cuda.is_available():
                self.nets = self.nets.cpu()
                outputs   = outputs.cpu()
            predictions = {}
            for k in range(0, len(self.nets)):            
                arch_str = list(self.samples)[k]
                predictions[arch_str] = outputs[1][k].detach().numpy().tolist()  # arch_str -> pred_label
            assert len(predictions) == len(self.nets) 
        else:
            predictions = self.pred_labels

        for k, v in predictions.items():
            # if v < self.sample_mean():
            if v[layer] == 1 :
                samples_badness[k] = self.samples[k]  # (val_loss, test_mae)
                samples_badness_labels[k] = predictions[k]
            else:
                samples_goodies[k] = self.samples[k]  # (val_loss, test_mae)
                samples_goodies_labels[k] = predictions[k]
        assert len(samples_badness) + len(samples_goodies) == len(self.samples)
        return samples_goodies, samples_badness, samples_goodies_labels, samples_badness_labels
