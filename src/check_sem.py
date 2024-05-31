import sys
import numpy as np
from tqdm import tqdm
import glob

from gensim.models import Word2Vec


def main():
    model_dir, save_dir = sys.argv[1:]
    for model_path in tqdm(sorted(glob.glob(f"{model_dir}/*"))):
        model = Word2Vec.load(model_path)
        d = model_path.split('_')[-3]  # e.g. "d100"
        print(model_path, f"{save_dir}/{d}_woman_king_man.npy", f"{save_dir}/{d}_dog.npy", flush=True)
        np.save(
            f"{save_dir}/{d}_woman_king_man.npy", 
            model.wv.most_similar(positive=['женщина', 'король'], negative=['мужчина'])
        )
        np.save(f"{save_dir}/{d}_dog.npy", model.wv.most_similar('собака'))

if __name__ == '__main__':
    main()
