import os
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict                

def get_embeddings(tokens, max_seq_length,\
                   embedding_type="word2vec",\
                   embedding_file_path="../resources/data/embeddings/model0.model",\
                   embedding_dim=300):
    # TODO: this is all embeddings, so for each language it should be an ndarray of dim = num functions x max seq length x embedding_dim, right?
    # Can be split into batches later?
    if embedding_dim is None or embedding_dim == 0:
        embedding_dim = 300

    if tokens is None or len(tokens) == 0:
        return None

    embedding_file_path = os.path.abspath(embedding_file_path)
    model = Word2Vec.load(embedding_file_path)
    embeddings = np.zeros((max_seq_length, embedding_dim))

    for j, token in enumerate(tokens):
        if embedding_type == "word2vec" and token not in model.wv.vocab:
            continue
        embedding = model.wv[token]
        embeddings[j,:] = embedding

    return embeddings