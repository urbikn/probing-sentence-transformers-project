import torch
import argparse
import os
from tqdm import tqdm
import src.embeddings as emb
import src.pca as pca

description = """
Main program that reproduced all the results from the report.

####################

It extract embeddings from two Sentence Transformers models:
- paraphrase-mpnet-base-v2
- paraphrase-MiniLM-L12-v2

The program has to be run in the following order:
    1. `python main.py -e`
        Extract the embeddings from the models and save them to disk in the src/.embeddings folder
    2. `python main.py`

####################
"""


def general_probing_args(parser):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument('--model', required=True, type=str, help='model name',
    #                     choices=['mpnet', 'minilm'])
    # parser.add_argument('--task', required=True, type=str, help='name of the task')
    # parser.add_argument('--seed', required=True, type=int, help='the random seed to check on')
    # parser.add_argument('--batch_size', type=int, help='batch size to train the probe', default=16)
    # parser.add_argument('--training_data', required=True, type=str,
    #                     help='path to the `.txt` dataset')
    parser.add_argument('--embedding_data',  type=str, default=None,
                        help='path to the `.pt` embeddings file of the same dataset.' + \
                        'Can be left blank and will take the dataset path and replace the file type')
    
    return parser

def extract_and_save_embeddings_bilstm(datasets):
    """
    Extract the embeddings of the baseline BiLSTM model and save them to disk.
    """
    # Test the BiLSTMEmbeddings class

    print(f'Generating BiLSTM embeddings')
    bilstm_embeddings = emb.BiLSTMEmbeddings(
        model_path='./.models/bilstm/model/infersent2.pkl',
        fasttext_path='./.models/bilstm/fastText/crawl-300d-2M.vec',
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

def reduce_embedding_dimensionality(embeddng_folder, target_dimension):
    """
    Reduce the dimensionality of the embeddings in the given folder to the given target dimension.
    """
    pca_instance = pca.initialise_PCA(target_dimension)

    # Go through each file in the folder that is a '.pt' file
    for file_name in tqdm(os.listdir(embeddng_folder), desc='Reducing dimensionality of embeddings'):
        if not file_name.endswith('.pt'):
            continue

        embeddings = emb.load_embeddings(f'{embeddng_folder}/{file_name}')
        reduced_embeddings = pca.reduce_dimensions(embeddings, pca_instance)
        torch.save(reduced_embeddings, f'{embeddng_folder}/{file_name.replace(".pt", "")}_pca.pt')

    return embedding_folder


if __name__ == '__main__':
    parser = general_probing_args(argparse.ArgumentParser())
    parser.add_argument('-e', '--extract', action='store_true', help='Extract the embeddings of the dataset from the models.')
    parser.add_argument('-p', '--pca', type=int, help='Reduce the dimensionality of the embeddings to the given number of dimensions.', default=0)
    args = parser.parse_args()

    # Define the datasets to be used
    datasets = ['bigram_shift', 'coordination_inversion', 'obj_number', 'odd_man_out', 'sentence_length', 'subj_number', 'top_constituents', 'tree_depth', 'word_content', 'past_present']

    embedding_folder = args.embedding_data

    # Check if the embeddings should be extracted
    if args.extract:
        extract_and_save_embeddings_bilstm(datasets)
        embedding_folder = extract_and_save_embeddings_sbert(datasets)
    if args.pca > 0 and embedding_folder is not None:
        if not os.path.exists(embedding_folder):
            raise ValueError('The embeddings folder does not exist. Path given: ' + embedding_folder)
        
        embedding_folder = reduce_embedding_dimensionality(embedding_folder, args.pca)