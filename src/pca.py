import torch, argparse
import numpy as np
from sklearn.decomposition import PCA
from .dataset import load_embeddings, save_embeddings, load_dataset

def initialise_PCA(n_components=100):
    pca = PCA(n_components)
    return pca

def reduce_dimensions(embedding_dict, pca):
    keys = list(embedding_dict.keys())
    if isinstance(embedding_dict[keys[0]], torch.Tensor):
        embeddings = np.array([embedding_dict[key].cpu() for key in keys])
    else:
        embeddings = np.array([embedding_dict[key] for key in keys])

    # Check if the embeddings are 2D
    if len(embeddings) > 2 and embeddings.shape[1] == 1:
        embeddings = embeddings.squeeze(1)
    
    pca.fit(embeddings)
    embedding_pca = pca.transform(embeddings)

    embedding_pca_dict = {key: embedding_pca[i] for i, key in enumerate(keys)}

    return embedding_pca_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reduce dimensions of embeddings.')
    parser.add_argument('task', type=str, help='Name of the current task.')
    args = vars(parser.parse_args())

    pca_inst = initialise_PCA(100)
    dataset = load_dataset('../.embeddings/{}.txt'.format(args['task']))
    embeddings = load_embeddings('../.embeddings/bilstm.{}.pt'.format(args['task']))
    reduced_embeddings = reduce_dimensions(embeddings, pca_inst)
    torch.save(reduced_embeddings, '../.embeddings/bilstm.{}_pca.pt'.format(args['task']))