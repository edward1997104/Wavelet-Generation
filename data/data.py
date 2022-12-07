import torch
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class WaveletSamples(torch.utils.data.Dataset):
    def __init__(self,
                 interval: int = 1,
                 load_ram = False,
                 data_files=None,
                 first_k = None):
        super(WaveletSamples, self).__init__()

        ### get file
        if data_files is None:
            data_files = []

        ## load the data folder
        self.data_preloaded = [ np.load(data_file[0]) for data_file in data_files ]


        if first_k is not None:
            self.data_path = self.data_path[:first_k]

        ## interval
        self.interval = interval

        ### data length
        self.data_len = self.data_preloaded[0].shape[0]



    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        idx = idx * self.interval

        processed_data = tuple([data[idx] for data in self.data_preloaded])

        return processed_data, idx
