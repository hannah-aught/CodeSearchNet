import os
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from models.model import get_data_files_from_directory
from collections import defaultdict

def load_raw_data(data_dirs, max_files_per_dir=None, parallelize=True):
    raw_samples = defaultdict(list)

    for data_dir in data_dir:
        files = get_data_files_from_directory(data_dirs, max_files_per_dir)
        for f in files:
            raw_sample = {}
            for line in f.read_by_file_suffix():
                language = line['language']

                if language.startswith('python'):
                    language = 'python'
                
                raw_sample['']
                raw_samples[language].append(raw_sample)

                

def get_embeddings(data, max_seq_length,\
                   embedding_type="word2vec",\
                   embedding_file_path="../resources/data/embeddings/model0.model",\
                   embedding_dim=300):
    # TODO: this is all embeddings, so for each language it should be an ndarray of dim = num functions x max seq length x embedding_dim, right?
    # Can be split into batches later?
    embedding_file_path = os.path.abspath(embedding_file_path)
    model = Word2Vec.load(embedding_file_path)
    
    for language, lines in data.items():
        for i, line in enumerate(lines):
            for key in ['code_tokens_func_name_as_query', 'code_tokens_docstring_as_query']:
                embeddings = np.zeros((max_seq_length,embedding_dim))
                for j, token in enumerate(line[key]):
                    if embedding_type == "word2vec" and token not in model.wv.vocab:
                        continue
                    embedding = model.wv[token]
                    embeddings[j,:] = embedding
                data[language][i][key] = embeddings