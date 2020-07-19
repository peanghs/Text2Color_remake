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

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :, :self.c_dim]
        logvar = x[:, :, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        std_size = std.size()
        #여기까지
        if torch.device == 'cpu':
            eps = torch.FloatTensor.normal_
        else:
            eps = torch.cuda.FloatTensor(std.size()).
        return eps * std + mu





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

        #
        return c_code, hidden, mu, logvar

