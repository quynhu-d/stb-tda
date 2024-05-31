import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

from gensim.models import Word2Vec


def train_cbow(
        dimension=2, 
        file_path='E:/nina_misc/RuPreprocessedNoSep.csv', 
        save_dir='E:/nina_misc/cbow_models'
):
    for chunk in tqdm(pd.read_csv(file_path, chunksize=1000), total=11):
        documents = chunk.preprocessed_text_no_sep.apply(lambda x: x.split()).tolist()
        if chunk.index[0] == 0:
            model = Word2Vec(sentences=documents, vector_size=dimension, min_count=3, workers=4)
        else:
            model = Word2Vec.load(f'{save_dir}/10k_cbow_d{dimension}_saved_model')
        model.build_vocab(documents, update=True)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(f'{save_dir}/10k_cbow_d{dimension}_saved_model')


def main():
    d, file_path, save_dir = sys.argv[1:]
    d = int(d)
    print(d, file_path, save_dir, sep='\n', flush=True)
    train_cbow(d, file_path, save_dir)


if __name__ == '__main__':
    main()
