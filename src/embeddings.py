import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import urllib.request
import shutil

from .bilstm import model as bilstm
from .dataset import load_dataset, save_embeddings, load_embeddings

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
        self.model.build_vocab_k_words(K=200000)  # Load embeddings of K most frequent words

        self.model.to(self.device)

    
    def embed(self, sentences, batch_size=32):
        """
        Embeds the sentences using the BiLSTM model.

        Outputs a numpy array of shape (len(sentences), 4096).
        """
        embeddings = self.model.encode(sentences, bsize=batch_size, tokenize=False, verbose=True)

        return embeddings

class SBERTEmbeddings:
    def __init__(self, model_path, device='cpu', output_hidden_states=False, cache_dir='./models/sbert'):
        self.device = device
        self.load(model_path, cache_dir=cache_dir, output_hidden_states=output_hidden_states)


    def load(self, model_path, output_hidden_states=False, cache_dir='./models/sbert'):
        """
        Loads the pretrained Sentence BERT model and tokenizer.

        The additional `output_hidden_states` is used to return the hidden
        representations from all layers of the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir, output_hidden_states=output_hidden_states).to(device=self.device)

    def mean_pooling(self, model_output, attention_mask):
        """
        Construct sentence embeddings from the hidden states of the model by using mean pooling.

        Outputs a numpy array of shape (1, 384).
        """
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    

    def embed(self, sentences):
        """
        Embeds the sentences using the SentenceBert model.

        Outputs a numpy array of shape (len(sentences), 384).
        """

        inputs = self.tokenizer(sentences, return_tensors='pt', truncation=True, padding=True)

        sentence_embeddings = []
        with torch.no_grad():
            # Get the embeddings for each sentence
            for i in tqdm(range(0, len(inputs['input_ids']))):
                input = inputs["input_ids"][i].unsqueeze(0).to(self.device)
                attention_mask = inputs["attention_mask"][i].unsqueeze(0).to(self.device)

                model_output = self.model(input, attention_mask=attention_mask)

                # Check if the model returns hidden states from all layers
                if model_output.hidden_states is not None:
                    model_embeddings = []

                    # Iterate over each hidden state in the model output and compute sentence embeddings
                    for hidden_state in model_output.hidden_states:
                        pooled_state = self.mean_pooling(hidden_state, attention_mask)
                        model_embeddings.append(pooled_state.cpu().detach())

                    # Stack all embeddings for the current input and add to the main list
                    stacked_embeddings = torch.stack(model_embeddings)
                    sentence_embeddings.append(stacked_embeddings)
                else:
                    # Compute the mean pooling for the model output and add to the main list
                    pooled_output = self.mean_pooling(model_output[0], attention_mask)
                    sentence_embeddings.append(pooled_output.cpu().detach())

                torch.cuda.empty_cache()

        # Concatenate all embeddings along axis 0
        sentence_embeddings = torch.stack(sentence_embeddings)

        return sentence_embeddings


if __name__ == "__main__":
    # Load an example dataset
    dataset_path = '../data/probing_data/subj_number.txt'

    # Limit to only 10 instances
    dataset = load_dataset(dataset_path)
    sentences = dataset['sentence'].values.tolist()

    # Test the BiLSTMEmbeddings class
    bilstm_embeddings = BiLSTMEmbeddings(
        model_path='./.models/bilstm/model/infersent2.pkl',
        fasttext_path='./.models/bilstm/fastText/crawl-300d-2M.vec',
        cache_dir='./.models/bilstm',
        device='cuda'
    )
    embeddings = bilstm_embeddings.embed(sentences)
    save_embeddings(
        embeddings,
        dataset,
        './.embeddings/bilstm.subj_number.pt'
    )

    exit

    # Test the SBERTEmbeddings class
    sbert_embeddings = SBERTEmbeddings(
        'sentence-transformers/paraphrase-MiniLM-L12-v2',
        output_hidden_states=True,
        cache_dir='./.models/sbert',
        device='cuda'
    )

    embeddings = sbert_embeddings.embed(sentences)

    for layer in range(embeddings.shape[1]):
        embedding = embeddings[:, layer, :, :]
        save_embeddings(
            embedding,
            dataset,
            f'./.embeddings/extended/minilm-layer-{layer}.subj_number.pt'
        )