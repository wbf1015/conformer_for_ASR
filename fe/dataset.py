import os
import numpy as np
from torch.utils.data import dataset
from .fbank import readFBankData

class AIshellData(dataset.Dataset):

    def __init__(self, path):
        super(AIshellData, self).__init__()
        self.base_path = path

    def __getitem__(self, index):
        file_list = os.listdir(self.base_path)
        file_name = file_list[index-1]
        file_path = self.base_path + file_name
        return readFBankData(file_path), file_name

    def __len__(self):
        file_list = os.listdir(self.base_path)
        return len(file_list)





if __name__ == '__main__':
    pass