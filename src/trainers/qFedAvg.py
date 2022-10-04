import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import time
from src.trainers.base import *


class Client(BaseClient):
    def __init__(self, id, params, dataset):
        super().__init__(id, params, dataset)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.optimizer = eval('optim.%s' % params['Trainer']['optimizer']['name'])(
            self.model.parameters(), 
            **params['Trainer']['optimizer']['params'],
        )
        self.params = params
        self.meters = {
            'accuracy': AvgMeter(),
            'classifier_loss': AvgMeter(), 
        }
    
    def calculate_loss(self):
        loss_meter = AvgMeter()
        with torch.no_grad():
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                classifier_loss = self.classifier_criterion(
                    outputs,
                    labels,
                )
                loss_meter.append(classifier_loss.item())
        return loss_meter.avg()
    
    def local_train(self):
        meters_classifier_loss = AvgMeter()
        omega = self.model.parameters_to_tensor().clone().detach()
        omega_loss = self.calculate_loss()
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
        with torch.no_grad():
            L = self.params['Trainer']['L']
            q = self.params['Trainer']['q']
            delta_omega = L * (omega - self.model.parameters_to_tensor())
            self.delta = delta_omega * (omega_loss ** q)
            self.h = q * (omega_loss ** (q - 1)) * (torch.norm(self.delta) ** 2) + L * (omega_loss ** q)
        self.meters['accuracy'].append(self.test_accuracy())
        self.meters['classifier_loss'].append(meters_classifier_loss.avg())

class Server(BaseServer):
    def aggregate_model(self, clients):
        n = len(clients)
        with torch.no_grad():
            omega_old = self.model.parameters_to_tensor()
            numerator = torch.zeros_like(omega_old)
            denominator = 0.0
            for client in clients:
                numerator += client.delta
                denominator += client.h
            omega = omega_old - numerator / denominator
        self.model.tensor_to_parameters(omega)
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
