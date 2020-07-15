import os
import torch
import pickle
import torch.utils.data as data
import numpy as np
import warnings
from skimage.color import rgb2lab

class PAT_Dataset(data.Dataset):
    # 데이터 준비
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
                # rgb를 lab 으로 변경 왜 d50(상관색 온도 5003K 일몰 이라함)을 썼는지는 모름
                # rgb2lab 이 2d는 못받아드리는듯 하여 폈다 다시 3개씩 잘라서 기록
                lab = rgb2lab(rgb[np.newaxis, np.newaxis, :]).flatten() # illuminant='d50' 일단 빼봄
                temp.append(lab[0])
                temp.append(lab[1])
                temp.append(lab[2])
            palette_list.append(temp)

        self.palette_seqs = torch.FloatTensor(palette_list)
        self.num_total_seqs = len(self.name_seqs)

    # Dataset 클래스의 필수 요소 : 로드한 data를 돌려줌
    def __getitem__(self, index):
        name_seq = self.name_seqs[index]
        palette_seq = self.palette_seqs[index]
        return name_seq, palette_seq

    # Dataset 클래스의 필수 요소 : 전체 데이터 길이 계산
    def __len__(self):
        return self.num_total_seqs



def t2p_loader(batch_size, input_dict):
    train_name_path = os.path.join('./data/hexcolor_vf/train_names.pkl')
    train_palette_path = os.path.join('./data/hexcolor_vf/train_palettes_rgb.pkl')
    test_name_path = os.path.join('./data/hexcolor_vf/test_names.pkl')
    test_palette_path = os.path.join('./data/hexcolor_vf/test_palettes_rgb.pkl')

    train_dataset = PAT_Dataset(train_name_path, train_palette_path, input_dict)
    # 코어 9900kf 는 8개라 반인 4개 사용, drop_last 완벽하지 않은 마지막 배치 버림)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4,
                                               drop_last=True, shuffle=True)
    test_dataset = PAT_Dataset(test_name_path, test_palette_path, input_dict)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4,
                                               drop_last=True, shuffle=False)
    return train_loader, test_loader
