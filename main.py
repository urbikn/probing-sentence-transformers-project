import torch
import numpy as np
import random
import json
import torch
import argparse
import os
import glob
from tqdm import tqdm
import src.embeddings as emb
import src.pca as pca
from src.dataset import ProbingDataset, collate_fn
from src.probe import MDLProbingClassifier


description = """
Main program that reproduced all the results from the report.

####################

It extract embeddings from two Sentence Transformers models:
- paraphrase-mpnet-base-v2
- paraphrase-MiniLM-L12-v2

The program has to be run in the following order:
    1. `python main.py -e --training_data data/probing_data/ --embedding_data .embeddings`
        Extract the embeddings from the models using the data from the data/probing_data 
        folder and save them to disk in the .embeddings folder
    2. `python main.py -r 100 --embedding_data .embeddings`
        Reduce the dimensionality of the embeddings in the .embeddings folder to 100 dimensions
    3. `python main.py
            -p 
            --model mpnet
            --task subj_number
            --seed 42
            --training_data data/probing_data/subj_number.txt
            --embedding_data .embeddings/subj_number.pt
            --save_report ./reports/probing/subj_number-mpnet.json
        `
        Run the probing experiment on the `subj_number` task/dataset using the embeddings from the `mpnet` model
        and save the results in the `./reports/probing/subj_number-mpnet.json` file.

Basically, to run everyting, run the following commands:
 main.py -e --training_data data/probing_data/ --embedding_data .embeddings && \    
 main.py -r 100 --embedding_data .embeddings && \
main.py -p --model mpnet --task subj_number --seed 42 --training_data data/probing_data/subj_number.txt --embedding_data .embeddings/subj_number.pt --save_report ./reports/probing/subj_number-mpnet.json && \

####################
"""


def general_probing_args(parser):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', type=str, help='(only when -p set) model name')
    parser.add_argument('--task', type=str, help='(only when -p set) name of the task')
    parser.add_argument('--seed', type=int, help='(only when -p set) the random seed to check on', default=42)
    parser.add_argument('--batch_size', type=int, help='(only when -p set) batch size to train the probe', default=16)
    parser.add_argument('--save_report',  type=str, help='(only when -p set) file where the report will be saved in `.json`')
    parser.add_argument('--training_data', type=str, default=None, help='path to the `.txt` dataset')
    parser.add_argument('--embedding_data',  type=str, default=None,
                        help='path to the `.pt` embeddings file of the same dataset.' + \
                        'Can be left blank and will take the dataset path and replace the file type')
    
    return parser

def extract_and_save_embeddings_bilstm(datasets):
    """
    Extract the embeddings of the baseline BiLSTM model and save them to disk.

    params:
        datasets: list of dataset names to extract the embeddings from
    
    returns:
        folder: the folder path where the embeddings were saved
    """
    # Test the BiLSTMEmbeddings class

    print(f'Generating BiLSTM embeddings')

    bilstm_embeddings = emb.BiLSTMEmbeddings(
        cache_dir='./.models/bilstm',
        device='cuda'
    )

    folder = './.embeddings'

    # Go through each dataset and extract the embeddings from the model
    for dataset_name in datasets:
        print('Loading dataset:', dataset_name)
            # Limit to only 10 instances
        dataset = emb.load_dataset(f'data/probing_data/{dataset_name}.txt')
        sentences = dataset['sentence'].values.tolist()

        embeddings = bilstm_embeddings.embed(sentences)
        emb.save_embeddings(
            embeddings,
            dataset,
            f'{folder}/bilstm.{dataset_name}.pt'
        )

        torch.cuda.empty_cache()
    
    return folder


def extract_and_save_embeddings_sbert(datasets):
    """
    Extract the embeddings of all layers from the two Sentence Transformers models and save them to disk.

    params:
        datasets: list of dataset names to extract the embeddings from

    returns:
        folder: the folder path where the embeddings were saved
    """
    # Define the models to be used to extract embeddings
    models = [
        'sentence-transformers/paraphrase-MiniLM-L12-v2',
        'sentence-transformers/paraphrase-mpnet-base-v2'
    ]

    folder = './.embeddings'

    # Go through each model
    for model_name in models:
        model_name_short = model_name.split('/')[-1]
        model = emb.SBERTEmbeddings(
            model_name,
            output_hidden_states=True,
            cache_dir='./.models/sbert',
            device='cuda'
        )
        print(f'Generating {model_name_short} embeddings')

        # Go through each dataset and extract the embeddings from the model
        for dataset_name in datasets:
            print('Loading dataset:', dataset_name)
                # Limit to only 10 instances
            dataset = emb.load_dataset(f'data/probing_data/{dataset_name}.txt')
            sentences = dataset['sentence'].values.tolist()
            embeddings = model.embed(sentences).cpu().detach().numpy()

            # Extract the embeddings from each layer and save them to disk
            for layer in range(embeddings.shape[1]):
                embedding = embeddings[:, layer, :, :]
                emb.save_embeddings(
                    embedding,
                    dataset,
                    f'{folder}/{model_name_short}-layer-{layer}.{dataset_name}.pt'
                )

            torch.cuda.empty_cache()
    
    return folder

def reduce_embedding_dimensionality(embedding_folder, target_dimension):
    """
    Reduce the dimensionality of the embeddings in the given folder to the given target dimension.

    params:
        embedding_folder: the folder path where the embeddings are read and saved
        target_dimension: the target dimension to reduce the embeddings to

    returns:
        embedding_folder: the embedding folder path that was given
    """
    pca_instance = pca.initialise_PCA(target_dimension)

    # Go through each file in the folder that is a '.pt' file
    for file_name in tqdm(os.listdir(embedding_folder), desc='Reducing dimensionality of embeddings'):
        if not file_name.endswith('.pt'):
            continue

        embeddings = emb.load_embeddings(f'{embedding_folder}/{file_name}')
        reduced_embeddings = pca.reduce_dimensions(embeddings, pca_instance)
        torch.save(reduced_embeddings, f'{embedding_folder}/{file_name.replace(".pt", "")}_pca.pt')

    return embedding_folder


def set_seed(seed):
    """ Set the seed for all the random generators.
    
    params:
        seed: number to set the seed to
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_embedding_size(train_dataset):
    """ Get the size of the embeddings in the given dataset.

    params:
        train_dataset: the dataset to get the embedding size from (must contain the `embedding` key)
    
    returns:
        size: the size of the embedding dimension
    """
    if 'embedding' not in train_dataset[0]:
        raise ValueError('The dataset does not contain the `embedding` key. Please make sure that the dataset was created with the `save_embeddings` function.')
    embedding = train_dataset[0]['embedding']
    if len(embedding.shape) == 1:
        return embedding.shape[0]
    else:
        return embedding.shape[1]

def run_probing_method(seed, training_folder, embedding_folder, collate_fn, task, batch_size, save_report=None):
    """ This function runs a probing method, analyzing the given datasets and saving a report.

    params:
        seed: The seed to set for reproducibility
        training_folder: The path to the training folder
        embedding_folder: The path to the embedding folder
        collate_fn: The collating function
        task: The task for the MDLProbingClassifier
        batch_size: The batch size for the MDLProbingClassifier
        save_report: The path to save the report. If None, will save in the default folder.

    raises:
        ValueError: If the training or embedding folder does not exist.
    """
    set_seed(seed) # Set the seed

    # Validate if both files exist
    if not os.path.exists(training_folder):
        raise ValueError('The dataset file does not exist. Path given: ' + training_folder)
    if not os.path.exists(embedding_folder):
        raise ValueError('The embeddings file does not exist. Path given: ' + embedding_folder)

    # Define the ID of the training run
    ID = embedding_folder.split('/')[-1].split('.')[0]

    # Define train, val, test datasets
    splits = ['train', 'val', 'test']
    datasets = {split: ProbingDataset(training_folder, embedding_folder, split) for split in splits}

    # Get the embedding size of the dataset
    embedding_size = get_embedding_size(datasets['train'])

    # Define the MDL probe and run the analysis
    mld_probe = MDLProbingClassifier(embedding_size, datasets['train'].num_classes(), device='cuda', ID=ID)
    report = mld_probe.analyize(datasets['train'], datasets['val'], datasets['test'], collate_fn, task=task, batch_size=batch_size)

    # If the report file is not given, save it in the default folder
    if save_report is None:
        save_report = f'.probing_reports/{task}-{ID}.json'

        if not os.path.exists('.probing_reports'):
            os.mkdir('.probing_reports')

    with open(save_report, 'w') as f:
        json.dump(report, f, indent=4)


if __name__ == '__main__':
    parser = general_probing_args(argparse.ArgumentParser())
    parser.add_argument('-e', '--extract', action='store_true', help='Extract the embeddings of the dataset from the models. (requires --training_data and --embedding_data)')
    parser.add_argument('-r', '--reduce_pca', type=int, help='Reduce the dimensionality of the embeddings to the given number of dimensions. (requires --embedding_data)', default=0)
    parser.add_argument('-p', '--probe', action="store_true", help='Tell program to probe the embeddings.')
    args = parser.parse_args()

    training_folder = args.training_data
    embedding_folder = args.embedding_data

    # Check if the embeddings should be extracted
    if args.extract and training_folder is not None and embedding_folder is not None:
        if os.path.exists(training_folder):
            filepaths = glob.glob(f'{training_folder}/*.txt')
            datasets = [os.path.splitext(os.path.basename(filepath))[0] for filepath in filepaths]

            extract_and_save_embeddings_bilstm(datasets)
            embedding_folder = extract_and_save_embeddings_sbert(datasets)
    # Check if the embeddings should be reduced
    if args.reduce_pca > 0 and embedding_folder is not None:
        if not os.path.exists(embedding_folder):
            raise ValueError('The embeddings folder does not exist. Path given: ' + embedding_folder)
        
        embedding_folder = reduce_embedding_dimensionality(embedding_folder, args.reduce_pca)

    # Check if the probing should be run
    if args.probe:
        run_probing_method(args.seed, training_folder, embedding_folder, collate_fn, args.task, args.batch_size, args.save_report)