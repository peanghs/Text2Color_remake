import torch
import torch.nn as nn
import numpy as np
import warnings
from skimage.color import lab2rgb, rgb2lab
import csv
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================== 텍스트 임베딩(t2p) ======================== #

class Dictionary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"} # SOS 문장 시작, EOS 문장 끝
        self.new_word_index = 2 # SOS랑 EOS 다음이니 2 aka) n_word
        self.max_len = 0

    def index_elements(self, data):
        for element in data:
            self.max_len = len(data) if self.max_len < len(data) else self.max_len
            # 원문은 index_element로 함수 분리 되어 있는것 합침
            # 단어 사전을 만드는 것이고 데이터에 단어가 없으면 인덱스에 추가하는 것
            if element not in self.word2index: # 새로운 단어라면
                self.word2index[element] = self.new_word_index # 인덱스 번호
                self.word2count[element] = 1 # 새 단어니 카운트 1
                self.index2word[self.new_word_index] = element # 단어 추가
                self.new_word_index += 1
            else:
                self.word2count[element] += 1 # 구 단어니 횟수 +1

def load_pretrained_embedding(dictionary, embed_file, embed_dim):
    if embed_file is None:
        return None

    # fasttext 적용
    pretrained_embed ={}
    with open(embed_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            # 공백이 자꾸 들어가서 그냥 301으로 자름 원래는 [1:]
            entries = tokens[1:301]
            if word == '<unk>':
                continue
            if word == '879129': # fasttext 첫줄 그냥 넘기기
                continue
            pretrained_embed[word] = entries
        f.close()

    vocab_size = len(dictionary) + 2
    Pre_emb = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    none_word = []
    for word, index in dictionary.items():
        if word in pretrained_embed:
            Pre_emb[index, :] = pretrained_embed[word]
            n += 1
        else:
            none_word.append(word)

    print(f"Glove 임베딩에서 {n}/{vocab_size} 단어가 초기화 되었습니다")
    # glove 3892 fasttext 4510
    return Pre_emb, none_word # 안나온 단어 추가

class Embed(nn.Module):
    #T2P 에서 모델에서 임베딩 계산시 사용
    def __init__(self, vocab_size, embed_dim, Pre_emb, train_emb):
        super(Embed, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        if Pre_emb is not None:
            print("사전에 학습된 단어 임베딩을 사용합니다...")
            self.embed.weight = nn.Parameter(Pre_emb)

        if train_emb == False:
            print("사전 학습된 단어 임베딩이 없습니다...")
            # requires_grad : 텐서의 모든 연산을 미분함
            self.embed.requires_grad = False

    def forward(self, doc):
        # ------------ 임베딩 확인용 코드 ----------------
        # doc 비교용 doc_ori
        doc_ori = doc
        doc_line_1 = doc_ori[0, :]
        doc_line_2 = doc_ori[1, :]
        doc_line_2_1 = torch.zeros(16, dtype=torch.int64).to(device='cuda')
        doc_line_2_2 = torch.zeros(16, dtype=torch.int64).to(device='cuda')
        doc_line_2_3 = torch.zeros(16, dtype=torch.int64).to(device='cuda')
        doc_line_2_1[0] = doc_line_2[0]
        doc_line_2_2[0] = doc_line_2[1]
        doc_line_2_3[0] = doc_line_2[2]

        doc_line_1 = self.embed(doc_line_1)
        doc_line_2 = self.embed(doc_line_2)
        doc_line_2_1 = self.embed(doc_line_2_1)
        doc_line_2_2 = self.embed(doc_line_2_2)
        doc_line_2_3 = self.embed(doc_line_2_3)
        # 여기서 text_embeddings에서 찾은 인풋 내 단어별 임베딩 id가 임베딩 실제 값으로 변환
        # 0은 (-0.6999, 0.4942..)
        # 공산주의 (0.1214, -0.0377..)
        # 어(-0.6686, 0.6350) 두운 (-0.0978, 0.0171) 네온 (-0.0443, 0.3061)
        # doc(배치 수, 최대 단어 길이, 임베딩)(32, 16, 300) 에서
        # (0,0,300) ->  공산주의(나머지 0,1:16,300 은 0) (1,0,300) -> 어 (1,1,300) -> 두운
        # ------------ 임베딩 확인용 코드 끝 ----------------

        doc = self.embed(doc)
        return doc


# ============================= 데이터 구성(p2c) ============================= #

def process_image(image_data, batch_size, imsize):
    input = torch.zeros(batch_size, 1, imsize, imsize)
    labels = torch.zeros(batch_size, 2, imsize, imsize)
    images_np = image_data.numpy().transpose((0, 2, 3, 1))

    for k in range(batch_size):
        img_lab = rgb2lab(images_np[k], illuminant='D55') #d50
        img_l = img_lab[:, :, 0] / 100
        input[k] = torch.from_numpy(np.expand_dims(img_l, 0))

        img_a_scale = (img_lab[:, :, 1:2] + 88) / 185
        img_b_scale = (img_lab[:, :, 2:3] + 127) / 212

        img_ab_scale = np.concatenate((img_a_scale, img_b_scale), axis=2)
        labels[k] = torch.from_numpy(img_ab_scale.transpose((2, 0, 1)))
    return input, labels

def process_palette_ab(pal_data, batch_size):
    img_a_scale = (pal_data[:, :, 1:2] + 88) / 185
    img_b_scale = (pal_data[:, :, 2:3] + 127) / 212
    img_ab_scale = np.concatenate((img_a_scale, img_b_scale), axis=2)
    ab_for_global = torch.from_numpy(img_ab_scale).float()
    ab_for_global = ab_for_global.view(batch_size, 10).unsqueeze(2).unsqueeze(2)
    return ab_for_global

def process_palette_lab(pal_data, batch_size):
    img_l = pal_data[:, :, 0:1] / 100
    img_a_scale = (pal_data[:, :, 1:2] + 88) / 185
    img_b_scale = (pal_data[:, :, 2:3] + 127) / 212
    img_lab_scale = np.concatenate((img_l, img_a_scale, img_b_scale), axis=2)
    lab_for_global = torch.from_numpy(img_lab_scale).float()
    lab_for_global = lab_for_global.view(batch_size, 15).unsqueeze(2).unsqueeze(2)
    return lab_for_global

def process_global_ab(input_ab, batch_size, always_give_global_hint):
    X_hist = input_ab

    if always_give_global_hint:
        B_hist = torch.ones(batch_size, 1, 1, 1)
    else:
        B_hist = torch.round(torch.rand(batch_size, 1, 1, 1))
        for l in range(batch_size):
            if B_hist[l].numpy() == 0:
                X_hist[l] = torch.rand(10)

    global_input = torch.cat([X_hist, B_hist], 1)
    return global_input

def process_global_lab(input_lab, batch_size, always_give_global_hint):
    X_hist = input_lab

    if always_give_global_hint:
        B_hist = torch.ones(batch_size, 1, 1, 1)
    else:
        B_hist = torch.round(torch.rand(batch_size, 1, 1, 1))
        for l in range(batch_size):
            if B_hist[l].numpy() == 0:
                X_hist[l] = torch.rand(15)

    global_input = torch.cat([X_hist, B_hist], 1)
    return global_input


# ============================= 기타 ============================= #

def init_weights_normal(m):
    if type(m) == nn.Conv1d:
        m.weight.data.normal_(0.0, 0.05)
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.05)

def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def lab2rgb_1d(in_lab, clip=True):
    warnings.filterwarnings("ignore")

    tmp_rgb = lab2rgb(in_lab[np.newaxis, np.newaxis, :], illuminant='D55').flatten() # 위에서 지웠으므로.. illuminant='D50'
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    return tmp_rgb

def write_txt(list, fname) :
    none_word_list = []
    print('-----없는 단어 저장 리스트-----')
    with open(fname, 'w', encoding='UTF-8') as file:
        for n, a in enumerate(list):
            a = a + "\n"
            file.write(a)
            none_word_list.append(a)
    print(none_word_list)
    print('-----없는 단어 저장 완료-----')
