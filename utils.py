import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).glob('*.npz'))

    def __getitem__(self, key):
        return self.read(self.files[key])

    def __iter__(self):
        yield from map(lambda file: self.read(file), self.files)

    def __len__(self):
        return len(self.files)

    def read(self, filepath):
        loader = np.load(filepath)

        X = loader['X']
        index = loader['index']
        columns = loader['columns']
        y = loader['y']

        df = pd.DataFrame(data=X, index=index, columns=columns)
        df['target'] = y
        return df
