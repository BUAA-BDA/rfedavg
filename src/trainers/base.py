import torch
import random
import importlib
import numpy as np
import torch.optim as optim
import time
import sys
import yaml
from tqdm import tqdm
from src.trainers.utils import *


class BaseClient():
    def __init__(self, id, params, dataset):
        self.batch_size = params['Trainer']['batch_size']
        self.trainset = dataset['train']
        self.testset = dataset['test']
        self.id = id
        collate_fn = None
        dataset_type = 'Image'
        if 'type' in params['Dataset'] and params['Dataset']['type'] == 'NLP':
            dataset_type = 'NLP'
        if dataset_type == 'NLP':
            collate_fn = nlp_collate_fn
        if self.trainset != None:
            self.trainloader = torch.utils.data.DataLoader(
                self.trainset, 
                batch_size=self.batch_size, 
                drop_last=True, 
                shuffle=True,
                collate_fn=collate_fn,
            )
        if self.testset != None:
            self.testloader = torch.utils.data.DataLoader(
                self.testset, 
                batch_size=self.batch_size, 
                drop_last=False, 
                shuffle=True,
                collate_fn=collate_fn,
            )
        self.E = params['Trainer']['E']
        self.device = torch.device(params['Trainer']['device'])
        models = importlib.import_module('src.models')
        self.model = eval('models.%s' % params['Model']['name'])(params)
        if dataset_type == 'NLP':
            self.model.embedding.weight.data.copy_(dataset['vocab'].vectors)
        self.model = self.model.to(self.device)
        self.optimizer = eval('optim.%s' % params['Trainer']['optimizer']['name'])(
            self.model.parameters(), 
            **params['Trainer']['optimizer']['params'],
        )
    
    def local_train(self):
        raise NotImplementedError()
    
    def clone_model(self, target):
        p_tensor = target.model.parameters_to_tensor()
        self.model.tensor_to_parameters(p_tensor)
        return

    def test_accuracy(self):
        if self.testset == None: return -1
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    
    def get_features_and_labels(self, train=True, batch=-1):
        dataloader = None
        if train: dataloader = self.trainloader
        else: dataloader = self.testloader
        features_batch = []
        labels_batch = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if i == batch: break
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                _, f_s = self.model(inputs, features=True)
                features_batch.append(f_s)
                labels_batch.append(labels)
        features = torch.cat(features_batch)
        labels = torch.cat(labels_batch)
        return features, labels
    
    def save_features_and_labels(self, fn, train=True, batch=-1):
        features, labels = self.get_features_and_labels(train, batch)
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        np.save('%s_features.npy' % fn, features)
        np.save('%s_labels.npy' % fn, labels)
        return

class BaseServer(BaseClient):
    def __init__(self, id, params, dataset):
        super().__init__(id, params, dataset)
        self.n_clients = params['Trainer']['n_clients']
        self.n_clients_per_round = round(params['Trainer']['C'] * self.n_clients)
        self.learning_rate = params['Trainer']['optimizer']['params']['lr']
        self.params = params

    def aggregate_model(self):
        raise NotImplementedError()

    def train(self):
        # finish 1 comm round
        raise NotImplementedError()

    def sample_client(self):
        return random.sample(
            self.clients, 
            self.n_clients_per_round,
        )

class Trainer():
    def __init__(self, config):
        # set seed
        set_seed(config['Trainer']['seed'])
        # import module
        trainer_module = importlib.import_module(
            'src.trainers.%s' % config['Trainer']['name']
        )
        dataset_module = importlib.import_module(
            'src.data.%s' % config['Dataset']['name']
        )
        # init meters
        self.meters = {
            'accuracy': AvgMeter(),
            'clients': {},
        }
        # get dataset
        dataset_func = eval('dataset_module.%s' % config['Dataset']['divide'])
        dataset_split, testset = dataset_func(config)
        # init clients
        self.clients = []
        for i in range(config['Trainer']['n_clients']):
            id = i + 1
            client=eval('trainer_module.Client')(
                       id, 
                       config,
                       dataset_split[i],
                   )
            self.clients.append(client)
        # init server
        self.server = eval('trainer_module.Server')(0, config, testset)
        self.server.clients = self.clients
        # save config
        self.config = config
    
    def train(self):
        output = sys.stdout
        if 'Output' in self.config: output = open(self.config['Output'], 'a')
        output.write(yaml.dump(self.config, Dumper=yaml.Dumper))
        try:
            for round in tqdm(range(self.config['Trainer']['Round']), desc='Communication Round', leave=False):
                output.write('==========Round %d begin==========\n' % round)
                time_begin = time.time()
                clients = self.server.train()
                self.meters['accuracy'].append(self.server.test_accuracy())
                time_end = time.time()
                for client in sorted(clients, key=lambda x: x.id):
                    client_summary = []
                    client_summary.append('client %d' % client.id)
                    for k, v in client.meters.items():
                        client_summary.append('%s: %.5f' % (k, v.last()))
                    output.write(', '.join(client_summary) + '\n')
                output.write('server, accuracy: %.5f\n' % self.meters['accuracy'].last())
                output.write('total time: %.0f seconds\n' % (time_end - time_begin))
                output.write('==========Round %d end==========\n' % round)
                output.flush()
        except KeyboardInterrupt:
            ...
        finally:
            acc_lst = self.meters['accuracy'].data
            avg_count = 5
            acc_avg = np.mean(acc_lst[-avg_count:])
            acc_std = np.std(acc_lst[-avg_count:])
            acc_max = np.max(acc_lst)
            output.write('==========Summary==========\n')
            for client in self.clients:
                client.clone_model(self.server)
                output.write('client %d, accuracy: %.5f\n' % (client.id, client.test_accuracy()))
            output.write('server, max accuracy: %.5f\n' % acc_max)
            output.write('server, final accuracy: %.5f +- %.5f\n' % (acc_avg, acc_std))
            output.write('===========================\n')
