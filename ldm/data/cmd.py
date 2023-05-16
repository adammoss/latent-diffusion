import os, yaml, pickle, shutil, tarfile, glob
import cv2
import requests
import albumentations
import PIL
import numpy as np
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import urllib


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


class CMDBase(Dataset):

    BASE_URL = "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/Maps_%s_LH_z=0.00.npy"

    def __init__(self, config=None, size=None, datasets=[]):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.cache_dir = self.config.get("cache_dir", "data")
        self._load(size=size, datasets=datasets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _load(self, data_size=None, size=None, norm_max=1.0, norm_min=-1.0,
              test_size=0.2, datasets=[], transform=np.log):
        data = []
        for i, dataset_name in enumerate(datasets):
            if not os.path.isfile(os.path.join(self.cache_dir, 'Maps_%s_LH_z=0.00.npy' % dataset_name)):
                url = self.BASE_URL % dataset_name
                download(url, os.path.join(self.cache_dir, "Maps_%s_LH_z=0.00.npy" % dataset_name))
            X = np.load(os.path.join(self.cache_dir, 'Maps_%s_LH_z=0.00.npy' % dataset_name)).astype(np.float32)
            if data_size is not None:
                X = X[0:data_size]
            if size is not None:
                X = np.array([resize(img, (size, size)) for img in X])
            X = transform(X)
            minimum = np.min(X, axis=0)
            maximum = np.max(X, axis=0)
            X = (X - minimum) / (maximum - minimum)
            X = (norm_max - norm_min) * X + norm_min
            X = np.expand_dims(X, -1)
            for j in range(len(X)):
                data.append({"image": X[j], "label": i})
        self.data_train, self.data_test = train_test_split(data, test_size=test_size, random_state=42)


class CMDTrain(CMDBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, i):
        return self.data_train[i]


class CMDValidation(CMDBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data_test)

    def __getitem__(self, i):
        return self.data_test[i]
