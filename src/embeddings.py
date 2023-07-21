import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

import os
import urllib.request
import shutil

import bilstm.model as bilstm

# usecase:
# -- Load the model that auto downloads or loads the pretrained model
# -- Extract the embeddings from the model

class BiLSTMEmbeddings:
    def __init__(self, model_path=None, fasttext_path=None, device='cpu', cache_dir='./models/bilstm'):
        self.device = device
        self.params = {
            'bsize': 64,
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'pool_type': 'max',
            'dpout_model': 0.0,
            'version': 2
        }

        # Check if both model_path and fasttext_path are provided or not provided
        if (model_path is not None and fasttext_path is None) or (model_path is None and fasttext_path is not None):
            raise ValueError("Please provide both 'model_path' and 'fasttext_path', or none to download.")
        
        # Download the model
        elif model_path is None:
            model_path, fasttext_path = self.download(cache_dir=cache_dir)

        # Load the model and the provided fasttext_path
        self.load(model_path, fasttext_path)


    def download(self, cache_dir=None):
        """
        Downloads the BiLSTM model pretrained on fastText.

        The data is accessible from the GitHub repo at https://github.com/facebookresearch/InferSent#use-our-sentence-encoder
        """
        print('Downloading fastText word vectors...')
        fasttext_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
        fasttext_path = os.path.join(cache_dir, "fastText")

        os.makedirs(fasttext_path, exist_ok=True)

        urllib.request.urlretrieve(fasttext_url, os.path.join(fasttext_path, os.path.basename(fasttext_url)))
        shutil.unpack_archive(os.path.join(fasttext_path, os.path.basename(fasttext_url)), fasttext_path)
        fasttext_path = os.path.join(fasttext_path, 'crawl-300d-2M.vec')
        
        print('Downloading BiLSTM...')
        model_url = 'https://dl.fbaipublicfiles.com/infersent/infersent2.pkl'
        model_path = os.path.join(cache_dir, 'model')

        os.makedirs(model_path, exist_ok=True)

        urllib.request.urlretrieve(model_url, os.path.join(model_path, os.path.basename(model_url)))
        model_path = os.path.join(model_path, 'infersent2.pkl')

        return model_path, fasttext_path
    
    def load(self, model_path, fasttext_path):
        """
        Loads the pretrained BiLSTM model and sets the w2v path.
        """
        self.model = bilstm.InferSent(self.params)
        self.model.load_state_dict(torch.load(model_path))
        self.model.set_w2v_path(fasttext_path)
        self.model.build_vocab_k_words(K=100000)  # Load embeddings of K most frequent words

        self.model.to(self.device)

    
    def embed(self, sentences):
        """
        Embeds the sentences using the BiLSTM model.
        """
        embeddings = self.model.encode(sentences, tokenize=False, verbose=True)

        return embeddings

class SBERTEmbeddings:
    def __init__(self, model_path, device='cpu', cache_dir='./models/sbert'):
        self.device = device
        self.load(model_path, cache_dir=cache_dir)


    def load(self, model_path, cache_dir='./models/sbert'):
        """
        Loads the pretrained BiLSTM model and sets the w2v path.
        """
        self.model = SentenceTransformer(model_path, cache_folder=cache_dir, device=self.device)
    

    def embed(self, sentences):
        """
        Embeds the sentences using the SentenceBert model.
        """
        embeddings = self.model.encode(sentences, show_progress_bar=True, device=self.device)

        return embeddings


def load_dataset(path):
    """
    Loads the dataset from the given path with the given format:
    <split_type>\t<label>\t<sentence>
    """
    dataset = pd.read_csv(path, sep='\t', header=None)
    dataset.columns = ['split_type', 'label', 'sentence']

    return dataset

def save_embeddings(embeddings, dataset, path):
    """
    Saves the embeddings as a dictionary <index, embedding> to the given path.
    """
    dict_embeddings = {}
    for i, embedding in enumerate(embeddings):
        dict_embeddings[i] = embedding

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(embeddings, path)


def load_embeddings(path):
    """
    Loads the embeddings from the given path.
    """
    return torch.load(path)


if __name__ == "__main__":

    # Load an example dataset
    dataset_path = '../data/probing_data/subj_number.txt'
    dataset = load_dataset(dataset_path)
    sentences = dataset['sentence'][:10].values.tolist()

    # Test the BiLSTMEmbeddings class
    # bilstm_embeddings = BiLSTMEmbeddings(cache_dir='./.models/bilstm')
    # embeddings = bilstm_embeddings.embed(sentences)
    # save_embeddings(embeddings, dataset, './.embeddings/bilstm.subj_number.pt')

    # Test the SBERTEmbeddings class
    sbert_embeddings = SBERTEmbeddings('sentence-transformers/paraphrase-MiniLM-L12-v2', cache_dir='./.models/sbert')
    embeddings = sbert_embeddings.embed(sentences)
    save_embeddings(embeddings, dataset, './.embeddings/sbert.subj_number.pt')