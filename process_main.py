import torch
import os
import pickle

from util import *


class main_solver(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_availavble() else 'cpu')
        print('---cuda mode---') if torch.cuda.is_available() else print('---cpu mode---')
        os.chdir('D:/Pycharm Project/Text2Color_remake') # 기본 경로 변경(상대경로 입력 시)
        self.build_model(args.mode)

    def build_model(self, mode):

        if mode == 'train_t2p':
            # 데이터 로드
            self.input_dict = self.prepare_dict()
            t2ploader



    def prepare_dict(self):
        input_dict = Dictionary()
        src_path = os.path.join('./data/hexcolor_vf/all_names.pkl')
        print(os.path.abspath(src_path))
        with open(src_path, 'rb') as f:
            text_data = pickle.load(f)
            f.close()

        print(f"--- {len(text_data)}개의 팔레트 이름을 불러오는 중입니다...")
        print("단어 사전을 만들고 있습니다...")

        for i in range(len(text_data)):
            input_dict.index_elements(text_data[i])
        return input_dict
