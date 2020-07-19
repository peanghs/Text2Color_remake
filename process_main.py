import torch
import os
import pickle

# from import는 그 내부 함수가 여러군데서 존재할 수 있기 때문
# util 과 data_loader의 함수명은 그래서 글로벌하게 유일해야 함
from util import *
from data_loader import *


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
            # 리턴이 train_loader 와 test_loader 이므로 test_loader는 받지 않음
            t2ploader, _ = t2p_loader(self.args.batch_size, self.input_dict)

            # 전이 학습할 Glove 임베딩 불러오기
            # 사전 학습된 데이터가 있는 경우 그걸 사용
            emb_file = os.path.join('./data/Color-Hex-vf.pth')
            if os.path.isfile(emb_file):
                Pre_emb = torch.load(emb_file)
            else:
                # 사전, 파일, 차원 순으로 호출해야 함
                Pre_emb = load_pretrained_embedding(self.input_dict.word2index,
                                                    './data/glove.840B.300d.txt', 300)
                Pre_emb = torch.from_numpy(Pre_emb)
                torch.save(Pre_emb, emb_file)
            Pre_emb = Pre_emb.to(self.device)

            # 생성기와 판별기 빌드
            self.encorder = T2P.EncoderRNN(self.input_dict.new_word_index, self.args.hidden_size,
                                           self.args.n_layers, self.args.dropout_p, Pre_emb).to(self.device)
            self.decoder = T2P.AttnDecoderRNN




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


    def train_t2p(self):
        pass
