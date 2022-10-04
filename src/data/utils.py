import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import Subset

def split_dataset_by_percent(train_dataset, test_dataset, s: float, num_user: int, func=(lambda x: x[1])):
    trainset_targets = [(i, func(item)) for i, item in enumerate(train_dataset)]
    testset_targets = [(i, func(item)) for i, item in enumerate(test_dataset)]
    random.shuffle(trainset_targets)
    random.shuffle(testset_targets)
    len_train_iid = round(s * len(train_dataset))
    len_test_iid = round(s * len(test_dataset))
    trainset_iid_idx = trainset_targets[:len_train_iid]
    trainset_niid_idx = trainset_targets[len_train_iid:]
    testset_iid_idx = testset_targets[:len_test_iid]
    testset_niid_idx = testset_targets[len_test_iid:]
    trainset_niid_idx = sorted(trainset_niid_idx, key=lambda x: x[1])
    testset_niid_idx = sorted(testset_niid_idx, key=lambda x: x[1])
    p_train_iid = 0
    p_train_niid = 0
    p_test_iid = 0
    p_test_niid = 0
    delta_train_iid = len(trainset_iid_idx) // num_user
    delta_train_niid = len(trainset_niid_idx) // num_user
    delta_test_iid = len(testset_iid_idx) // num_user
    delta_test_niid = len(testset_niid_idx) // num_user
    dataset_split = []
    for _ in range(num_user):
        train_idx = []
        test_idx = []
        if delta_train_iid > 0:
            train_idx.extend(
                trainset_iid_idx[
                    p_train_iid: p_train_iid + delta_train_iid
                ]
            )
        if delta_train_niid > 0:
            train_idx.extend(
                trainset_niid_idx[
                    p_train_niid: p_train_niid + delta_train_niid
                ]
            )
        if delta_test_iid > 0:
            test_idx.extend(
                testset_iid_idx[
                    p_test_iid: p_test_iid + delta_test_iid
                ]
            )
        if delta_test_niid > 0:
            test_idx.extend(
                testset_niid_idx[
                    p_test_niid: p_test_niid + delta_test_niid
                ]
            )
        train_idx = list(map(lambda x: x[0], train_idx))
        test_idx = list(map(lambda x: x[0], test_idx))
        dataset_split.append(
            {
                'train': Subset(train_dataset, train_idx),
                'test': Subset(test_dataset, test_idx),
            }
        )
        p_train_iid += delta_train_iid
        p_train_niid += delta_train_niid
        p_test_iid += delta_test_iid
        p_test_niid += delta_test_niid
    return dataset_split