import torch
import argparse
import os
import src.embeddings as emb

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

def parse_args():
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', required=True, type=str, help='the model type',
                        choices=["basic", "finetuning", "random"])
    parser.add_argument('--feature_extractor', default='roberta-base', type=str, choices=['roberta-base', 'deberta-base'],
                        help='model to use as a feature extractor')
    parser.add_argument('--training_data', required=False, type=str, help='the data type the model was trained on',
                        choices=["raw", "scrubbed"], default=None)
    parser.add_argument('--training_balanced', type=str, help='balancing of the training data',
                        choices=["subsampled", "oversampled", "original"], default="original")
    parser.add_argument('--type', type=str, help='the type of vectors to probe',
                        choices=["raw", "scrubbed"])
    parser.add_argument('--testing_balanced', type=str, help='balancing of the testing data',
                        choices=["subsampled", "oversampled", "original"], default="original")


    args = parser.parse_args()
    print(args)
    return args

def general_probing_args(parser):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', required=True, type=str, help='model name',
                        choices=['mpnet', 'minilm'])
    parser.add_argument('--task', required=True, type=str, help='name of the task')
    parser.add_argument('--seed', required=True, type=int, help='the random seed to check on')
    parser.add_argument('--batch_size', type=int, help='batch size to train the probe', default=16)
    parser.add_argument('--training_data', required=True, type=str,
                        help='path to the `.txt` dataset')
    parser.add_argument('--embedding_data',  type=str,
                        help='path to the `.pt` embeddings file of the same dataset.' + \
                        'Can be left blank and will thake the dataset path and replace the file type')

def extract_and_save_embeddings():
    """
    Extract the embeddings of all layers from the two Sentence Transformers models and save them to disk.
    """
    # Define the models to be used to extract embeddings
    models = [
        'sentence-transformers/paraphrase-MiniLM-L12-v2',
        'sentence-transformers/paraphrase-mpnet-base-v2'
    ]

    # Define the datasets to be used
    datasets = ['bigram_shift', 'coordination_inversion', 'obj_number', 'odd_man_out', 'sentence_length', 'subj_number', 'top_constituents', 'tree_depth', 'word_content']
    
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
                    f'./.embeddings/{model_name_short}-layer-{layer}.{dataset_name}.pt'
                )

            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-e', '--extract', action='store_true', help='Extract the embeddings of the dataset from the models.')
    args = parser.parse_args()

    # Check if the embeddings should be extracted
    if args.extract:
        extract_and_save_embeddings()