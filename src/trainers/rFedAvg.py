import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from src.trainers.base import *
from src.trainers.utils import LinearMMD


class Client(BaseClient):
    def __init__(self, id, params, dataset):
        super().__init__(id, params, dataset)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.mmd_criterion = LinearMMD()
        self.params = params
        self.meters = {
            'accuracy': AvgMeter(),
            'classifier_loss': AvgMeter(), 
            'mmd_loss': AvgMeter(),
            'loss': AvgMeter(),
        }
    
    def local_train(self):
        meters_classifier_loss = AvgMeter()
        meters_mmd_loss = AvgMeter()
        meters_loss = AvgMeter()
        for epoch in range(self.E):
            for i, data in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, f_s = self.model(inputs, features=True)
                classifier_loss = self.classifier_criterion(
                    outputs,
                    labels,
                )
                mmd_loss = sum([self.mmd_criterion(f_s, f_t) for f_t in self.f_t])
                mmd_loss /= len(self.f_t)
                loss = classifier_loss + mmd_loss * self.params['Trainer']['lambda']
                loss.backward()
                self.optimizer.step()
                meters_classifier_loss.append(classifier_loss.item())
                meters_mmd_loss.append(mmd_loss.item())
                meters_loss.append(loss.item())
        self.meters['accuracy'].append(self.test_accuracy())
        self.meters['classifier_loss'].append(meters_classifier_loss.avg())
        self.meters['mmd_loss'].append(meters_mmd_loss.avg())
        self.meters['loss'].append(meters_loss.avg())

    def get_features(self):
        inputs, _ = next(iter(self.trainloader))
        inputs = inputs.to(self.device)
        _, f_s = self.model(inputs, features=True)
        return f_s


class Server(BaseServer):
    def aggregate_model(self, clients):
        n = len(clients)
        p_tensors = []
        self.f_t.clear()
        for _, client in enumerate(clients):
            p_tensors.append(client.model.parameters_to_tensor())
        avg_tensor = sum(p_tensors) / n
        self.model.tensor_to_parameters(avg_tensor)
        return

    def train(self):
        # assign C < 1.0 is meaningless
        clients = self.sample_client()
        self.f_t = [client.get_features().detach() for client in clients]

        # for each client in choose_clients
        for i, client in enumerate(clients):
            client.f_t = [item for j, item in enumerate(self.f_t) if i != j]
            for p in client.optimizer.param_groups:
                p['lr'] = self.learning_rate
        
        for client in clients:
            client.clone_model(self)
            client.local_train()
        
        # aggregate params
        self.aggregate_model(clients)

        self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']
        
        return clients
