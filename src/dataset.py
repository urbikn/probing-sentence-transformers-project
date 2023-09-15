import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_dataset(path):
    """
    Loads the dataset from the given path with the given format:
    <split_type>\t<label>\t<sentence>

    or in case of probing data:
    <split_type>\t<label>\t<sentence>\t<uid>

    params:
        path: path to the dataset
    
    returns:
        dataset: pandas dataframe with the dataset
    """
    dataset = pd.read_csv(path, sep='\t', header=None)

    if len(dataset.columns) == 3:
        dataset.columns = ['split_type', 'label', 'sentence']
    else:
        dataset.columns = ['split_type', 'label', 'sentence', 'uid']

    return dataset

def load_embeddings(path):
    """
    Loads the embeddings from the given path.

    params:
        path: path to the embeddings file
    
    returns:
        embeddings: dict of embeddings with the UID as key
    """
    embedding = torch.load(path)
    return torch.load(path)

def save_embeddings(embeddings, dataset, path):
    """
    Saves embeddings and dataset to the given path, both aligned with a UID.

    Path format should be: <model_name>.<dataset_name>.pt

    params:
        embeddings: list of embedding vectors
        dataset: pandas dataframe with the dataset
        path: path to save the embeddings and dataset

    returns:
        None
    """
    if len(os.path.basename(path).split('.')) != 3:
        raise ValueError("Path format should be <model_name>.<dataset_name>.pt")

    # Create a dataset UID to map the embeddings to the dataset
    model_name, dataset_name, _ = os.path.basename(path).split('.')
    dataset_uid = [f'{dataset_name}-{index}' for index in range(len(dataset))]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the dataset with the UID
    dataset_copy = dataset.copy()
    dataset_copy['uid'] = dataset_uid
    dataset_copy.to_csv(os.path.join(os.path.dirname(path), f'{dataset_name}.txt'), sep='\t', index=False)

    # Save the embeddings
    dict_embeddings = dict(zip(dataset_uid, embeddings))
    torch.save(dict_embeddings, path)


class ProbingDataset(Dataset):
    """
    Dataset class for probing tasks that loads the dataset and embeddings from the given paths.

    dataset: <split_type>\t<label>\t<sentence>\t<uid>
    embeddings: <uid>: <embedding>

    Possible split types: 'train', 'val', 'test'
    """
    def __init__(self, dataset_path, embeddings_path, split_type='train'):
        split_types = {'train': 'tr', 'val': 'va', 'test': 'te'}

        self.data = load_dataset(dataset_path)
        self.label_map = {lbl: idx for idx, lbl in enumerate(self.data['label'].unique())}
        self.data = self.data[self.data['split_type'] == split_types[split_type]] # Splits based on its split type

        self.embeddings = load_embeddings(embeddings_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if isinstance(self.embeddings[row['uid']], np.ndarray):
            embedding = torch.tensor(self.embeddings[row['uid']])
        else:
            embedding = self.embeddings[row['uid']].clone().detach()
        label = torch.tensor(self.label_map[row['label']])

        return {
            'text': row['sentence'],
            'embedding': embedding,
            'label': label
        }

    
    def num_classes(self):
        return len(self.label_map)


def collate_fn(batch):
    """
    Collate function for the DataLoader that converts the batch to a dict of tensors.
    """

    # assume the batch is a list of dicts with 'text', 'embedding', and 'label' keys
    text_batch = [d['text'] for d in batch]
    # Convert embeddings 
    embedding_batch = torch.stack([d['embedding'] for d in batch])
    label_batch = torch.stack([d['label'] for d in batch])

    return {'text': text_batch, 'embedding': embedding_batch, 'label': label_batch}