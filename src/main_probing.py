import torch
import argparse
import json
import os
import pprint
import numpy as np
import random

from dataset import ProbingDataset, collate_fn
from probe import MDLProbingClassifier

description = """
Program that reproduced all the probing experiments

####################

An example of running the program when the embeddings are already extracted:
    `python main_probing.py 
        --model mpnet
        --task subj_number
        --seed 42
        --training_data ../data/probing_data/subj_number.txt
        --embedding_data ../data/probing_data/subj_number.pt
        --save_report ../reports/probing/subj_number-mpnet.json
    `
This will run the probing experiment on the `subj_number` task/dataset using the embeddings from the `mpnet` model.

The results will be saved in the `../reports/probing/subj_number-mpnet.json` file.
####################
"""

def general_probing_args():
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', required=True, type=str, help='model name',
                        choices=['mpnet', 'minilm'])
    parser.add_argument('--task', required=True, type=str, help='name of the task')
    parser.add_argument('--training_id', type=str, help='ID with which to identify the training run')
    parser.add_argument('--seed', required=True, type=int, help='the random seed to check on')
    parser.add_argument('--batch_size', type=int, help='batch size to train the probe', default=16)
    parser.add_argument('--training_data', required=True, type=str,
                        help='path to the `.txt` dataset')
    parser.add_argument('--embedding_data',  required=True, type=str,
                        help='path to the `.pt` embeddings file of the dataset dataset.')
    parser.add_argument('--save_report',  type=str,
                        help='file where the report will be saved in `.json`')

    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_embedding_size(train_dataset):
    if 'embedding' not in train_dataset[0]:
        raise ValueError('The dataset does not contain the `embedding` key. Please make sure that the dataset was created with the `save_embeddings` function.')
    return train_dataset[0]['embedding'].shape[1]

if __name__ == '__main__':
    args = general_probing_args()

    # Set the seed
    set_seed(args.seed)

    dataset_file= args.training_data
    embeddings_file = args.embedding_data

    # Validate if both files exist
    if not os.path.exists(dataset_file):
        raise ValueError('The dataset file does not exist. Path given: ' + dataset_file)
    if not os.path.exists(embeddings_file):
        raise ValueError('The embeddings file does not exist. Path given: ' + embeddings_file)

    # Define the ID of the training run (just so we can identify it later by the model, layer and task)
    if args.training_id is None:
        ID = embeddings_file.split('/')[-1].split('.')[0]
    else:
        ID = args.training_id

    # Define train, val, test datasets
    splits = ['train', 'val', 'test']
    datasets = {split: ProbingDataset(dataset_file, embeddings_file, split) for split in splits}

    embedding_size = get_embedding_size(datasets['train'])

    # Define the MDL probe and run the analysis
    mld_probe = MDLProbingClassifier(embedding_size, datasets['train'].num_classes(), device='cuda', ID=ID)
    report = mld_probe.analyize(datasets['train'], datasets['val'], datasets['test'], collate_fn, task=args.task, batch_size=args.batch_size)

    # == Save the report == #
    if args.save_report is None:
        args.save_report = f'.probing_reports/{args.task}-{ID}.json'

        if not os.path.exists('.probing_reports'):
            os.mkdir('.probing_reports')

    with open(args.save_report, 'w') as f:
        json.dump(report, f, indent=4)