import os
import torch
import pickle
import torch.utils.data as data
import numpy as np
import warnings

class PAT_Dataset(data.Dataset):
    def __init__(self, name_path, palette_path, input_dict):
        with open(name_path, 'rb') as fin:
            self.name_seqs = pickle.load(fin)
        with open(palette_path, 'rb') as fin:
            self.palette_seqs = pickle.load(fin)

        words_index = []
        for index, palette_name in enumerate(self.name_seqs):
            temp = [0] * input_dict.max_len

            for i, word in enumerate(palette_name):
                temp[i] = input_dict.word2index[word]
            words_index.append(temp)
        self.name_seqs = torch.LongTensor(words_index)

        palette_list = []
        for index, palettes in enumerate(self.palette_seqs):
            temp = []
            for palette in palettes:
                rgb = np.array([palette[0], palette[1], palette[2]]) / 255.0
                warnings.filterwarnings("ignore") # 경고 메시지 숨기기




def t2p_loader(batch_size, input_dict):
    train_name_path = os.path.join('./data/hexcolor_vf/train_names.pkl')
    train_palette_path = os.path.join('./data/hexcolor_vf/train_palettes_rgb.pkl')
    test_name_path = os.path.join('./data/hexcolor_vf/test_names.pkl')
    test_palette_path = os.path.join('./data/hexcolor_vf/test_palettes_rgb.pkl')

    train_dataset = PAT_Dataset(train_name_path, train_palette_path, input_dict)
    train_loader = data.DataLoader