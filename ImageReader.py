from matplotlib import pyplot as plt
from matplotlib import image as img
from os import listdir
import numpy as np
import collections

class ImageReader:
    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    @classmethod
    def read(cls, type = "train", onehot = True):
        l = []
        for id, name in enumerate(ImageReader.classes):
            l += ImageReader._read(f"./data/seg_{type}/seg_{type}/{name}/", id)
        name_list = [name for name, _, _ in l]
        image_list = np.stack([image for _, image, _ in l], axis = 0)
        id_list = [id for _, _, id in l]
        if onehot:
            id_onehot = np.zeros((len(id_list), 6))
            id_onehot[np.arange(len(id_list)), id_list] = 1  
            return name_list, image_list, id_onehot
        else:
            return name_list, image_list, np.array(id_list)
    @classmethod
    def _read(cls, path, id):
        _l = []
        for filename in listdir(path):
            image = img.imread(path + filename) / 255.0
            pad_width = 150 - image.shape[0]
            if pad_width:
                image = np.pad(image, ((0, pad_width), (0,0), (0,0)), mode = 'constant')
            _l += [(filename, image, id)]
        return _l