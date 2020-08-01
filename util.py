import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    pretrained_embed ={}
    with open(embed_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            entries = tokens[1:]
            if word == '<unk>':
                continue
            pretrained_embed[word] = entries
        f.close()

    vocab_size = len(dictionary) + 2
    Pre_emb = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for word, index in dictionary.items():
        if word in pretrained_embed:
            Pre_emb[index, :] = pretrained_embed[word]
            n += 1

    print(f"Glove 임베딩에서 {n}/{vocab_size} 단어가 초기화 되었습니다")
    return Pre_emb

class Embed(nn.Module):
    #T2P 에서 모델에서 임베딩 계산시 사용
    def __init__(self, vocab_size, embed_dim, Pre_emb, train_emb):
        super(Embed, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        if Pre_emb is not None:
            print("사전에 학습된 단어 임베딩을 사용합니다")
            self.embed.weight = nn.Parameter(Pre_emb)

        if train_emb == False:
            print("사전 학습된 단어 임베딩이 없습니다")
            # requires_grad : 텐서의 모든 연산을 미분함
            self.embed.requires_grad = False

    def forward(self, doc):
        # doc 에 임베딩 업데이트
        doc = self.embed(doc)
        return doc


