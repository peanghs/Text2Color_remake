import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dictionary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"} # SOS 문장 시작, EOS 문장 끝
        self.new_word_index = 2 # SOS랑 EOS 다음이니 2
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




