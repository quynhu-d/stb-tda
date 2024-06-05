import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
import glob


def get_word_space(model, filename):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = f.read()
    words = set([w for w in corpus.split() if w in model])
    # words = sorted(words)
    space = np.vstack([model[w] for w in words])
    return space
    # return space, words

def get_ngram_space(model, filename, n=2):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = f.read().split()
    ngrams = set()
    ngram_space = []
    for i in range(len(corpus) - n + 1):
        ngram = [corpus[i + j] for j in range(n)]
        flag_out = False
        for w in ngram: 
            if w == '.': 
                flag_out = True
                break
            if w not in model: 
                flag_out = True
                break
        if flag_out: continue
        if " ".join(ngram) in ngrams: continue
        ngram_space.append(np.hstack([model[w] for w in ngram]))
        ngrams.add(" ".join(ngram))
    ngram_space = np.array(ngram_space)
    # print(ngram_space.shape)
    return ngram_space    
    # return ngram_space, ngrams

from sklearn.metrics import pairwise_distances
def get_dist_to_centers_array(e, hole_e_centers):
    a = pairwise_distances(e, hole_e_centers, metric='cosine')
    return np.hstack([a, a.mean(axis=1).reshape(-1, 1)])

from collections import Counter
def get_most_common_closest_hole(min_dist):
    cnt = Counter(min_dist.argmin(axis=1))
    a = np.array([cnt[hn] for hn in range(min_dist.shape[1])]) / min_dist.shape[0]
    return np.hstack([a, [a.argmax()]])

def get_dist_array(e, hole_embs, apply_func=np.min):
    dist_list = np.vstack([apply_func(pairwise_distances(e, hole, metric='cosine'), axis=1) for hole in hole_embs])
    dist_list = np.vstack([dist_list, dist_list.mean(axis=0)])
    return dist_list.T

def process(dir_path, model, part = 'word', lang = 'RU'):
    """
        Get features for dataset.

        Parameters:
            dir_path - files of which type to use
            model - CBoW model to retrieve embeddings
            part - "word"/"bigram"/"trigram", level of analysis
            lang - "EN"/"RU"
        
        Returns:
            features - list of features for each text
            text_names - list of text names
    """
    # data_part = 'Train'
    # text_type = 'lit'
    # dir_path = "../DATASET/Russian/{data_part}/{text_type}/*.txt"
    files = sorted(glob.glob(dir_path))
    print(len(files), flush=True)
    files = files[:3]

    features = []
    text_names = []
    hole_embeddings = np.load(f"holes/{lang.upper()}/{part}s/hole_embeddings.npy", allow_pickle=True).item()
    hole_e_centers = np.vstack([h.mean(axis=0) for h in hole_embeddings.values()])
    for f in tqdm(files):
        word_space = get_word_space(model, f)
        c_mean = np.mean(get_dist_to_centers_array(word_space, hole_e_centers), axis=0)
        m1 = get_dist_array(word_space, hole_embeddings.values(), np.min)
        m2_mean = np.mean(get_dist_array(word_space, hole_embeddings.values(), np.max), axis=0)
        h = get_most_common_closest_hole(m1[:,:-1])
        features.append(np.hstack([c_mean, np.mean(m1, axis=0), m2_mean, h]))
        text_names.append(f.split('_')[-1][:-4])

    return features, text_names    