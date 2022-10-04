import pandas as pd
import torch
from torchtext.vocab import Vocab, Counter
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.functional import (
    totensor, 
    vocab_func, 
    sequential_transforms
)
from torchtext.experimental.datasets.text_classification import TextClassificationDataset

# data load
df = pd.read_csv(
    "./data/training.1600000.processed.noemoticon.csv", 
    engine="python", 
    header=None,
    encoding='ISO-8859-1',
)
df[0] = df[0].replace(to_replace=4, value=1)

def get_vocab_counter(data, transforms):
    counter = Counter()
    for line in data:
        for tokens in transforms(line):
            counter[tokens] += 1
    return counter

def niid_device(params):
    num_user = params['Trainer']['n_clients']
    dataset_user = params['Dataset']['user']
    assert num_user == dataset_user # should be exact same
    usernames = list(dict(df[4].value_counts()))[:dataset_user]
    df_small = df.loc[df[4].isin(usernames)]
    df_small = df_small.sample(frac=1) # shuffle all the data
    df_train = df_small.iloc[:int(df_small.shape[0] * 0.9), :]
    df_test = df_small.iloc[int(df_small.shape[0] * 0.9):, :]
    text_transform = sequential_transforms(
        str.lower, 
        get_tokenizer("basic_english"),
    )
    counter = Counter(dict(
        get_vocab_counter(df_train[5], text_transform).most_common(3000 - 2)
    ))
    vocab = Vocab(
        counter, 
        vectors='glove.6B.300d', 
        vectors_cache='./data/vector_cache/',
    )
    text_transform = sequential_transforms(
        text_transform, 
        vocab_func(vocab), 
        totensor(dtype=torch.long), 
    )
    label_transform = sequential_transforms(totensor(dtype=torch.long))
    data_test = list(zip(df_test[0], df_test[5]))
    test_dataset = TextClassificationDataset(
        data_test, 
        vocab, 
        (label_transform, text_transform),
    )
    # pandas is easy to split
    #data_train = list(zip(df_train[0], df_train[5]))
    #train_dataset = TextClassificationDataset(data_train, vocab, (label_transform, text_transform))
    dataset_split = []
    for username in usernames:
        split_train = df_small.loc[df_small[4] == username]
        split_train = list(zip(split_train[0], split_train[5]))
        dataset_split.append(
            {
                'train': TextClassificationDataset(
                    split_train, 
                    vocab, 
                    (label_transform, text_transform),
                ),
                'test': None, 
            }
        )
    for item in dataset_split: item['vocab'] = vocab
    testset_dict = {
        'train': None,
        'test': test_dataset,
        'vocab': vocab,
    }
    return dataset_split, testset_dict

def iid_device(params):
    num_user = params['Trainer']['n_clients']
    dataset_user = params['Dataset']['user']
    usernames = list(dict(df[4].value_counts()))[:dataset_user]
    df_small = df.loc[df[4].isin(usernames)]
    df_small = df_small.sample(frac=1) # shuffle all the data
    df_train = df_small.iloc[:int(df_small.shape[0] * 0.9), :]
    df_test = df_small.iloc[int(df_small.shape[0] * 0.9):, :]
    text_transform = sequential_transforms(
        str.lower, 
        get_tokenizer("basic_english"),
    )
    counter = Counter(dict(
        get_vocab_counter(df_train[5], text_transform).most_common(3000 - 2)
    ))
    vocab = Vocab(
        counter, 
        vectors='glove.6B.300d', 
        vectors_cache='./data/vector_cache/',
    )
    text_transform = sequential_transforms(
        text_transform, 
        vocab_func(vocab), 
        totensor(dtype=torch.long), 
    )
    label_transform = sequential_transforms(totensor(dtype=torch.long))
    data_test = list(zip(df_test[0], df_test[5]))
    test_dataset = TextClassificationDataset(
        data_test, 
        vocab, 
        (label_transform, text_transform),
    )
    # pandas is easy to split
    #data_train = list(zip(df_train[0], df_train[5]))
    #train_dataset = TextClassificationDataset(data_train, vocab, (label_transform, text_transform))
    df_train_iid = df_train.sample(frac=1)
    p_train_iid = 0
    delta_train_iid = df_train_iid.shape[0] // num_user
    dataset_split = []
    for userid in range(num_user):
        train_lst = []
        split_train = df_train_iid[
            p_train_iid: p_train_iid + delta_train_iid
        ]
        split_train = list(zip(split_train[0], split_train[5]))
        dataset_split.append(
            {
                'train': TextClassificationDataset(
                    split_train, 
                    vocab, 
                    (label_transform, text_transform),
                ),
                'test': None, 
            }
        )
        p_train_iid += delta_train_iid
    for item in dataset_split: item['vocab'] = vocab
    testset_dict = {
        'train': None,
        'test': test_dataset,
        'vocab': vocab,
    }
    return dataset_split, testset_dict

def niid(params):
    num_user = params['Trainer']['n_clients']
    dataset_frac = params['Dataset']['frac']
    s = params['Dataset']['s']
    df_small = df.sample(frac=dataset_frac) # sample & shuffle
    df_train = df_small.iloc[:int(df_small.shape[0] * 0.9), :]
    df_test = df_small.iloc[int(df_small.shape[0] * 0.9):, :]
    text_transform = sequential_transforms(
        str.lower, 
        get_tokenizer("basic_english"),
    )
    counter = Counter(dict(
        get_vocab_counter(df_train[5], text_transform).most_common(3000 - 2)
    ))
    vocab = Vocab(
        counter, 
        vectors='glove.6B.300d', 
        vectors_cache='./data/vector_cache/',
    )
    text_transform = sequential_transforms(
        text_transform, 
        vocab_func(vocab), 
        totensor(dtype=torch.long), 
    )
    label_transform = sequential_transforms(totensor(dtype=torch.long))
    data_test = list(zip(df_test[0], df_test[5]))
    test_dataset = TextClassificationDataset(
        data_test, 
        vocab, 
        (label_transform, text_transform),
    )
    # pandas is easy to split
    #data_train = list(zip(df_train[0], df_train[5]))
    #train_dataset = TextClassificationDataset(data_train, vocab, (label_transform, text_transform))
    df_train_iid = df_train.iloc[:int(s * df_train.shape[0]), :]
    df_train_niid = df_train.iloc[int(s * df_train.shape[0]):, :].sort_values([0])
    p_train_iid = 0
    p_train_niid = 0
    delta_train_iid = df_train_iid.shape[0] // num_user
    delta_train_niid = df_train_niid.shape[0] // num_user
    dataset_split = []
    for userid in range(num_user):
        train_lst = []
        if delta_train_iid > 0:
            train_lst.append(
                df_train_iid[
                    p_train_iid: p_train_iid + delta_train_iid
                ]
            )
        if delta_train_niid > 0:
            train_lst.append(
                df_train_niid[
                    p_train_niid: p_train_niid + delta_train_niid
                ]
            )
        split_train = pd.concat(train_lst)
        split_train = list(zip(split_train[0], split_train[5]))
        dataset_split.append(
            {
                'train': TextClassificationDataset(
                    split_train, 
                    vocab, 
                    (label_transform, text_transform),
                ),
                'test': None, 
            }
        )
        p_train_iid += delta_train_iid
        p_train_niid += delta_train_niid
    for item in dataset_split: item['vocab'] = vocab
    testset_dict = {
        'train': None,
        'test': test_dataset,
        'vocab': vocab,
    }
    return dataset_split, testset_dict