import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


import glob
def get_holes_per_chunk(dir="holes/RU/words", filenames="ru_word_holes_*.npy"):
    """
        Get all homology representatives (cycles) for each chunk.
    """
    hole_contours = {}
    for file in glob.glob(f"{dir}/{filenames}"):
        hole_contours["_".join(file.split('_')[3:])[:-4]] = get_cycles(np.load(file) - 1)
    return hole_contours

def collect_holes_from_chunks(dir="holes/RU/words/"):
    """
        Merge all homologies from different chunks.
    """
    def get_homology_birth_death(files):
        h_bd = {}
        for fn in files:
            chunk_num = "_".join(fn.split('_')[1:])[:-4]
            h_bd[chunk_num] = np.load(fn)
        h_all = np.hstack(list(h_bd.values()))
        h_lifetimes = np.diff(h_all, axis=0)
        return h_all, h_lifetimes, h_bd
    
    h0_all, h0_lifetimes, h0_bd = get_homology_birth_death(glob.glob(f"{dir}/h0_*.npy"))
    h1_all, h1_lifetimes, h1_bd = get_homology_birth_death(glob.glob(f"{dir}/h1_*.npy"))
    plt.scatter(*h0_all, s=.5, label='$H_0$');
    plt.scatter(*h1_all, s=.5, label='$H_1$');
    plt.legend();
    plt.xlabel('birth')
    plt.ylabel('death')
    plt.show()
    return (h0_all, h0_lifetimes, h0_bd), (h1_all, h1_lifetimes, h1_bd)


def get_cycles(cycles):
    """
        Get homology representatives (cycles) based on list of connected edges.
    """
    new_cycle_flag = True
    cycles_list = []
    cur_cycle = set()
    for a, b in cycles:
        if new_cycle_flag:
            cur_cycle.update([a])
            cur_cycle.update([b])
            cycles_list.append([])
            new_cycle_flag = False
            continue
        if a in cur_cycle:
            cur_cycle.remove(a)
            cycles_list[-1].append(a)
        else:
            cur_cycle.update([a])
        if b in cur_cycle:
            cur_cycle.remove(b)
            cycles_list[-1].append(b)
        else:
            cur_cycle.update([b])
        if len(cur_cycle) == 0:
            new_cycle_flag = True
    return cycles_list

def get_hole_knn(model, words):
    """
        Returns matrix with number of k for each pair of words using w2v model.
    """
    def all_words_in_topn(words, topn):
        for w in words:
            if w not in topn:
                return False
        return True
    
    hole_knn = []    
    for w1 in tqdm(words, leave=False):
        n = 1000
        topn = [_[0] for _ in model.wv.most_similar(w1, topn=n)]
        other_words = [w2 for w2 in words if w1 != w2]
        while not all_words_in_topn(other_words, topn):
            n += 1000
            topn = [_[0] for _ in model.wv.most_similar(w1, topn=n)]
        l = []
        for w2 in words:
            if w1 == w2: l.append(0)
            else:
                l.append(topn.index(w2) + 1)
        hole_knn.append(l)
    return pd.DataFrame(np.array(hole_knn), columns=words, index=words)