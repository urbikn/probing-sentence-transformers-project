import torch, argparse
import numpy as np
from sklearn.decomposition import PCA
from .dataset import load_embeddings, save_embeddings, load_dataset

def initialise_PCA(n_components=100):
    r""" Initialise the PCA instance with the given number of components.

    params:
        n_components: dimension number to reduce to
    
    returns:
        pca: PCA instance from sklearn
    """
    pca = PCA(n_components)
    return pca

def reduce_dimensions(embedding_dict, pca):
    r""" 
    This function reduces the dimensionality of the given embeddings using PCA.

    Parameters:
        embedding_dict: A dictionary where the keys are words and the values are their corresponding embeddings.
        pca: A PCA object from sklearn.decomposition.PCA.

    Returns:
        embedding_pca_dict: A dictionary where the keys are words and the values are their reduced-dimensionality embeddings.
    """
    keys = list(embedding_dict.keys())

    # Check if the embeddings are torch Tensors. If so, convert them to numpy arrays.
    if isinstance(embedding_dict[keys[0]], torch.Tensor):
        embeddings = np.array([embedding_dict[key].cpu() for key in keys])
    else:
        embeddings = np.array([embedding_dict[key] for key in keys])

    # Check if the embeddings are 2D. If they are, remove the extra dimension.
    if len(embeddings.shape) > 2 and embeddings.shape[1] == 1:
        embeddings = embeddings.squeeze(1)
    
    # Fit the PCA on the embeddings and then transform the embeddings
    pca.fit(embeddings)
    embedding_pca = pca.transform(embeddings)
    embedding_pca_dict = {key: embedding_pca[i] for i, key in enumerate(keys)}

    return embedding_pca_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reduce dimensions of embeddings.')
    parser.add_argument('task', type=str, help='Name of the current task.')
    args = vars(parser.parse_args())

    pca_inst = initialise_PCA(100)
    dataset = load_dataset('.embeddings/{}.txt'.format(args['task']))
    embeddings = load_embeddings('.embeddings/bilstm.{}.pt'.format(args['task']))
    reduced_embeddings = reduce_dimensions(embeddings, pca_inst)
    torch.save(reduced_embeddings, '.embeddings/bilstm.{}_pca.pt'.format(args['task']))