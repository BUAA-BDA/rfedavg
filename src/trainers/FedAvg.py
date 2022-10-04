import torch
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
from src.trainers.base import *


class Client(BaseClient):
    def __init__(self, id, params, dataset):
        super().__init__(id, params, dataset)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.params = params
        self.meters = {
            'accuracy': AvgMeter(),
            'classifier_loss': AvgMeter(),
        }
    
    def local_train(self):
        meters_classifier_loss = AvgMeter()
        for epoch in range(self.E):
            for i, data in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                classifier_loss = self.classifier_criterion(
                    outputs,
                    labels,
                )
                classifier_loss.backward()
                self.optimizer.step()
                meters_classifier_loss.append(classifier_loss.item())
        self.meters['accuracy'].append(self.test_accuracy())
        self.meters['classifier_loss'].append(meters_classifier_loss.avg())

class Server(BaseServer):
    def aggregate_model(self, clients):
        n = len(clients)
        p_tensors = []
        for _, client in enumerate(clients):
            p_tensors.append(client.model.parameters_to_tensor())
        avg_tensor = sum(p_tensors) / n
        self.model.tensor_to_parameters(avg_tensor)
        return

    def train(self):
        # random clients
        clients = self.sample_client()

        for client in clients:
            # send params
            client.clone_model(self)
            for p in client.optimizer.param_groups:
                p['lr'] = self.learning_rate
        
        for client in clients:
            # local train
            client.local_train()
        
        # aggregate params
        self.aggregate_model(clients)

        self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']
        
        return clients
