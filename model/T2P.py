import torch
import torch.nn as nn
from util import *

class CA_NET(nn.Module):

    def __init__(self):
        super(CA_NET, self).__init__()
        # 인풋 차원
        self.t_dim = 150
        # 아웃풋 차원
        self.c_dim = 150
        self.fc = nn.Linear(self.t_dim, self.c_dim*2, bias=True)
        self.relu = nn.ReLU()

    # VAE 오토 인코더 특징 추출
    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :, :self.c_dim]
        logvar = x[:, :, :self.c_dim] # self.c_dim: 다른 이유를 모르겠음
        return mu, logvar

    def reparametrize(self, mu, logvar):
        # https://ratsgo.github.io/generative%20model/2018/01/27/VAE/
        std = logvar.mul(0.5).exp_()
        std_size = std.size()
        # eps = torch.cuda.FloatTensor(std.size()).normal_(0.0, 1)
        if torch.device == 'cpu':
            eps = torch.FloatTensor(std.size()).normal_(0.0, 1) # 여기 하다 맘
        else:
            eps = torch.cuda.FloatTensor(std.size()).normal_(0.0, 1)
        return eps * std + mu

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class EncoderRNN(nn.Module):
    # 왜인진 모르나 Pre_emb = None 으로 되어 있음, 일단 None 삭제함
    def __init__(self, input_size, hidden_size, n_layer, dropout_p, Pre_emb):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layer
        # util 에서 Embed 불러옴
        # input vocab_size, embed_dim, Pre_emb, train_emb
        self.embed = Embed(input_size, 300, Pre_emb, True)
        # input input_size, hidden_size, num_layer
        self.gru = nn.GRU(300, hidden_size, n_layer, dropout=dropout_p)
        # 위에 있음
        self.ca_net = CA_NET()

    def forward(self, word_inputs, hidden):
        embedded = self.embed(word_inputs).transpose(0,1)
        output, hidden = self.gru(embedded, hidden)
        # text_embedding 에 output이 들어감
        c_code, mu, logvar = self.ca_net(output)

        return c_code, hidden, mu, logvar

    def init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, input_dict, hidden_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.input_dict = input_dict
        # 아래 클래스에서 불러옮
        self.attn = Attn(hidden_size, input_dict.max_len)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.palette_dim = 3

        self.gru = nn.GRUCell(self.hidden_size + self.palette_dim, hidden_size)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLu(inplace=True),
            nn.BatchNorm1d(hidden_size), nn.Linear(hidden_size, self.palette_dim))

    def forward(self, last_palette, last_decoder_hidden, encoder_outputs, each_input_size, i):

        if i == 0:
            context = torch.mean(encoder_outputs, dim=1, keepdim=True)
            # cat 1 은 추가 행이 아니라 뒤로 잇는 것..
            # 그래서 last_palette랑 context 가 뒤에서 2번째 모양이 같아야 함
            gru_input = torch.cat((last_palette, context.squeeze(1)), 1)
            gru_hidden = self.gru(gru_input, last_decoder_hidden)

            palette = self.out(gru_hidden.squeeze(0))
            return palette, context.unsqueeze(0), gru_hidden

        else:
            attn_weights = self.attn(last_decoder_hidden.squeeze(0), encoder_outputs, each_input_size)
            context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))

            gru_input = torch.cat((last_palette, context.squeeze(1)), 1)
            gru_hidden = self.gru(gru_input, last_decoder_hidden)

            palette = self.out(gru_hidden.squeeze(0))
            return palette, context.unsqueeze(0), gru_hidden, attn_weights


class Attn(nn.Module):
    def __init__(self, hidden_size, max_length):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=0)
        self.attn_e = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_energey = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden, encoder_output, each_size):
        seq_len = encoder_output.size(0)
        batch_size = encoder_output.size(1)
        #cpu 환경 일단 지원
        attn_energies = torch.zeros(seq_len, batch_size, 1).cuda() if torch.device == 'cuda' else \
            torch.zeros(seq_len, batch_size, 1)

        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_output[i])

        attn_energies = self.softmax(attn_energies)
        return attn_energies.permute(1, 2, 0) # 순서 바꿈, batch_size, 1, seq_len

    def score(self, hidden, encoder_output):
        # 뒤의 _는 기존에 파이썬 명령어가 있을 때 충돌을 피하기 위해 사용
        encoder_ = self.attn_e(encoder_output)
        hidden_ = self.attn_h(hidden)
        energy = self.attn_energey(self.sigmoid(encoder_ + hidden_))

        return energy



