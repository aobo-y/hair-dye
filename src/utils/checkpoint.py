import os
import pickle
import torch

META_FILE = 'meta.pt'

class CheckpointManager:
    def __init__(self, path):
        self.path = path

    def __fullpath(self, name):
        return os.path.join(self.path, name)

    def __ensure_folder(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, name, data):
        filepath = self.__fullpath(name + '.tar')
        self.__ensure_folder()
        torch.save(data, filepath)

    def save_meta(self, **kargs):
        self.__ensure_folder()
        with open(self.__fullpath(META_FILE), 'wb') as fout:
            pickle.dump(kargs, fout)

    def purge(self, *args):
        pass

    def load(self, name, device):
        filepath = self.__fullpath(name + '.tar')
        checkpoint = torch.load(filepath, map_location=device)

        if not os.path.exists(self.__fullpath(META_FILE)):
            return checkpoint

        with open(self.__fullpath(META_FILE), 'rb') as fin:
            meta = pickle.load(fin)

        return {**checkpoint, **meta}

    def loadMeta(self):
        pass

    def save_image(self, name, fig):
        filepath = self.__fullpath(name + '.png')
        self.__ensure_folder()

        fig.savefig(filepath)


