import torch
import os
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as ticker
from random import randint
from util_kor import palette_diversity
from image_util import change_color, change_color_hsv, change_color_lch



# from import는 그 내부 함수가 여러군데서 존재할 수 있기 때문
# util 과 data_loader의 함수명은 그래서 글로벌하게 유일해야 함
from util import *
from data_loader import *
from model import T2P, P2C


class main_solver(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'---------------{self.device} mode---------------')
        os.chdir('D:/Pycharm Project/Text2Color_remake') # 기본 경로 변경(상대경로 입력 시)
        self.build_model(args.mode)

    def prepare_dict(self):
        input_dict = Dictionary()
        if self.args.language == 'kor_ft':
            src_path = os.path.join('./data/hexcolor_vf/kor_all_names_okt.pkl')
        elif self.args.language == 'kor_gl':
            src_path = os.path.join('./data/hexcolor_vf/kor_all_names_mecab.pkl')
        else:
            src_path = os.path.join('./data/hexcolor_vf/all_names.pkl')
        print(os.path.abspath(src_path))
        with open(src_path, 'rb') as f:
            text_data = pickle.load(f)
            f.close()

        print(f"--- {len(text_data)}개의 팔레트 이름을 불러오는 중입니다...")
        print("단어 사전을 만들고 있습니다...")

        for i in range(len(text_data)):
            input_dict.index_elements(text_data[i])
        print(f"단어 사전에서 {len(input_dict.word2index)}개의 단어를 추출했습니다...")
        return input_dict

    def prepare_data(self, images, palettes, always_give_global_hint, add_L):
        # p2c 에서 사용
        batch = images.size(0)
        imsize = images.size(3)

        inputs, labels = process_image(images, batch, imsize)
        if add_L:
            for_global = process_palette_lab(palettes, batch)
            global_hint = process_global_lab(for_global, batch, always_give_global_hint)
        else:
            for_global = process_palette_ab(palettes, batch)
            global_hint = process_global_ab(for_global, batch, always_give_global_hint)

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        global_hint = (global_hint).expand(-1, -1, imsize, imsize).to(self.device)
        return inputs, labels, global_hint


    def build_model(self, mode):

        if mode == 'train_t2p':
            # 데이터 로드
            self.input_dict = self.prepare_dict()
            # 리턴이 train_loader 와 test_loader 이므로 test_loader는 받지 않음
            self.train_loader, _ = t2p_loader(self.args.batch_size, self.input_dict, self.args.language)

            # 전이 학습할 Glove 임베딩 불러오기
            # 사전 학습된 데이터가 있는 경우 그걸 사용
            if self.args.language == 'kor_ft':
                emb_file = os.path.join('./data/Color-Hex-vf-kr-ft.pth')
            elif self.args.language == 'kor_gl':
                emb_file = os.path.join('./data/Color-Hex-vf-kr-gl.pth')
            else:
                emb_file = os.path.join('./data/Color-Hex-vf.pth')
            if os.path.isfile(emb_file):
                Pre_emb = torch.load(emb_file)
            else:
                # 사전, 파일, 차원 순으로 호출해야 함
                if self.args.language == 'kor_ft':
                    # fasttext 썼는데 905 단어만 걸려서 기존것으로 대체 // 기존은 1041/
                    Pre_emb, none_word = load_pretrained_embedding(self.input_dict.word2index,
                                                                   './data/wiki.ko.vec', 300)
                elif self.args.language == 'kor_gl':
                    # fasttext 썼는데 905 단어만 걸려서 기존것으로 대체 // 기존은 1041/
                    Pre_emb, none_word = load_pretrained_embedding(self.input_dict.word2index,
                                                                   './data/glove_kor.txt', 100)
                else:
                    Pre_emb, none_word = load_pretrained_embedding(self.input_dict.word2index,
                                                                   './data/glove.840B.300d.txt', 300)
                # 없는 단어 로그 파일 저장
                nw_log_time = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
                nw_log_path = self.args.lab_dir + nw_log_time + "_none_word_" + self.args.language + ".txt"
                write_txt(none_word,nw_log_path)

                Pre_emb = torch.from_numpy(Pre_emb)
                torch.save(Pre_emb, emb_file)
            Pre_emb = Pre_emb.to(self.device)

            # 생성기와 판별기 빌드
            self.encoder = T2P.EncoderRNN(self.input_dict.new_word_index, self.args.hidden_size,
                                           self.args.n_layers, self.args.dropout_p, Pre_emb, self.args.language).to(self.device)
            self.decoder = T2P.AttnDecoderRNN(self.input_dict, self.args.hidden_size,
                                              self.args.n_layers, self.args.dropout_p).to(self.device)
            # 15는 color_size
            self.discriminator = T2P.Discriminator(15, self.args.hidden_size).to(self.device)

            # weight 초기화 // init_weights_normal은 util 에 있음
            self.encoder.apply(init_weights_normal)
            self.decoder.apply(init_weights_normal)
            self.discriminator.apply(init_weights_normal)

            # 옵티마이저
            self.decoder_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
            self.decoder_optimizer = torch.optim.Adam(self.decoder_parameters,
                                                lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))

        elif mode == 'test_t2p':
            # 데이터 불러오기
            self.input_dict = self.prepare_dict()

            # 글로브 임베딩 호출
            if self.args.language == 'kor_ft':
                emb_file = os.path.join('./data', 'Color-Hex-vf-kr-ft.pth')
            elif self.args.language == 'kor_gl':
                emb_file = os.path.join('./data', 'Color-Hex-vf-kr-gl.pth')
            else:
                emb_file = os.path.join('./data', 'Color-Hex-vf.pth')
            if os.path.isfile(emb_file):
                Pre_emb = torch.load(emb_file)
            else:
                if self.args.language == 'kor_ft':
                    Pre_emb, none_word = load_pretrained_embedding(self.input_dict.word2index, './data/wiki.ko.vec', 300)
                elif self.args.language == 'kor_gl':
                    Pre_emb, none_word = load_pretrained_embedding(self.input_dict.word2index, './data/glove_kor.txt', 100)
                else:
                    Pre_emb, none_word = load_pretrained_embedding(self.input_dict.word2index, './data/glove.840B.300d.txt', 300)
                Pre_emb = torch.from_numpy(Pre_emb)

                # 없는 단어 로그 파일 저장
                nw_log_time = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
                nw_log_path = self.args.lab_dir + nw_log_time + "_none_word_" + self.args.language + ".txt"
                write_txt(none_word,nw_log_path)

                # 모델 저장
                torch.save(Pre_emb, emb_file)

            Pre_emb = Pre_emb.to(self.device)

            # 데이터 호출
            self.test_loader, self.imsize = test_loader(self.args.dataset, self.args.batch_size, self.input_dict,
                                                        self.args.language)
            # 학습된 제너레이터 호출
            self.encoder = T2P.EncoderRNN(self.input_dict.new_word_index, self.args.hidden_size,
                                      self.args.n_layers, self.args.dropout_p, Pre_emb, self.args.language).to(self.device)
            self.decoder_T2P = T2P.AttnDecoderRNN(self.input_dict, self.args.hidden_size,
                                        self.args.n_layers, self.args.dropout_p).to(self.device)

        elif mode == 'train_p2c':
            # 데이터 불러오기
            self.train_loader, imsize = p2c_loader(self.args.dataset, self.args.batch_size, 0)

            # 생성기와 판별기 빌드
            self.decoder = P2C.UNet(imsize, self.args.add_L).to(self.device)
            self.discriminator = P2C.Discriminator(self.args.add_L, imsize).to(self.device)

            # 옵티마이저
            self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.args.lr,
                                                      weight_decay=self.args.weight_decay)
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                                            weight_decay=self.args.weight_decay)
            self.decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, 'min',
                                                                                patience=5, factor=0.1)
            self.discriminator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.discriminator_optimizer,
                                                                                      'min', patience=5, factor=0.1)

    def load_model(self, mode, resume_epoch):
        if mode == 'train_t2p':
            if self.args.language == 'kor_ft':
                encoder_path = os.path.join(self.args.t2p_dir, '{}_G_encoder_kor_ft.ckpt'.format(resume_epoch))
                decoder_path = os.path.join(self.args.t2p_dir, '{}_G_decoder_kor_ft.ckpt'.format(resume_epoch))
                discriminator_path = os.path.join(self.args.t2p_dir, '{}_D_kor_ft.ckpt'.format(resume_epoch))
            elif self.args.language == 'kor_gl':
                encoder_path = os.path.join(self.args.t2p_dir, '{}_G_encoder_kor_gl.ckpt'.format(resume_epoch))
                decoder_path = os.path.join(self.args.t2p_dir, '{}_G_decoder_kor_gl.ckpt'.format(resume_epoch))
                discriminator_path = os.path.join(self.args.t2p_dir, '{}_D_kor_gl.ckpt'.format(resume_epoch))
            else:
                encoder_path = os.path.join(self.args.t2p_dir, '{}_G_encoder.ckpt'.format(resume_epoch))
                decoder_path = os.path.join(self.args.t2p_dir, '{}_G_decoder.ckpt'.format(resume_epoch))
                discriminator_path = os.path.join(self.args.t2p_dir, '{}_D.ckpt'.format(resume_epoch))
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=lambda storage,
                                                                                                  loc: storage))

        elif mode == 'test_t2p':
            print(f'{format(resume_epoch)} 에폭의 모델로 테스트를 시작합니다...')
            if self.args.language == 'kor_ft':
                encoder_path = os.path.join(self.args.t2p_dir, '{}_G_encoder_kor_ft.ckpt'.format(resume_epoch))
                decoder_path = os.path.join(self.args.t2p_dir, '{}_G_decoder_kor_ft.ckpt'.format(resume_epoch))
            elif self.args.language == 'kor_gl':
                encoder_path = os.path.join(self.args.t2p_dir, '{}_G_encoder_kor_gl.ckpt'.format(resume_epoch))
                decoder_path = os.path.join(self.args.t2p_dir, '{}_G_decoder_kor_gl.ckpt'.format(resume_epoch))
            else:
                encoder_path = os.path.join(self.args.t2p_dir, '{}_G_encoder.ckpt'.format(resume_epoch))
                decoder_path = os.path.join(self.args.t2p_dir, '{}_G_decoder.ckpt'.format(resume_epoch))
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
            self.decoder_T2P.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

        elif mode == 'train_p2c':
            decoder_path = os.path.join(self.args.p2c_dir, '{}_G.ckpt'.format(resume_epoch))
            discriminator_path = os.path.join(self.args.p2c_dir, '{}_D.ckpt'.format(resume_epoch))
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=lambda storage,
                                                                                                  loc: storage))



    def train_t2p(self):
        # matplotlib 한글 폰트 설정
        if self.args.language == 'kor_ft' or 'kor_gl':
            font_fname = 'C:/Windows/Fonts/MALGUNSL.ttf'
            font_family = font_manager.FontProperties(fname=font_fname).get_name()
            plt.rcParams["font.family"] = font_family

        # loss function
        criterion_GAN = nn.BCELoss()
        criterion_smoothL1 = nn.SmoothL1Loss()

        # 학습 시작
        start_epoch = 0
        if self.args.resume_epoch:
            print(f'모델을 불러오고 {format(self.args.resume_epoch)} 에폭부터 학습을 시작합니다...')
            start_epoch = self.args.resume_epoch
            self.load_model(self.args.mode, self.args.resume_epoch)
        else:
            print('신규 학습을 시작합니다...')

        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        #시작 시간 저장
        start_time = time.time()
        for epoch in range(start_epoch, self.args.num_epochs):
            for batch_idx, (txt_embeddings, real_palettes) in enumerate(self.train_loader):

                # 텍스트 입력 사이즈 계산
                batch_size = txt_embeddings.size(0)
                nonzero_indices = list(torch.nonzero(txt_embeddings)[:, 0])
                each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]

                # 트레이닝 데이터 준비.
                txt_embeddings = txt_embeddings.to(self.device)
                real_palettes = real_palettes.to(self.device).float()

                # BCE loss 를 위한 라벨 준비.
                real_labels = torch.ones(batch_size).to(self.device)
                fake_labels = torch.zeros(batch_size).to(self.device)

                # 입력 출력 변수 설정.
                palette = torch.FloatTensor(batch_size, 3).zero_().to(self.device)
                # 왜 15지?
                fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(self.device)

                # 인코더 변수 설정.
                encoder_hidden = self.encoder.init_hidden(batch_size).to(self.device)
                encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings, encoder_hidden)

                # 팔레트 생성.
                for i in range(5):
                    palette, decoder_context, decoder_hidden, _ = self.decoder(palette, decoder_hidden.squeeze(0),
                                                                               encoder_outputs,each_input_size,i)
                    fake_palettes[:, 3 * i:3 * (i + 1)] = palette

                # 판별기 설정.
                each_input_size = torch.FloatTensor(each_input_size).to(self.device)
                each_input_size = each_input_size.unsqueeze(1).expand(batch_size, self.decoder.hidden_size)
                encoder_outputs = torch.sum(encoder_outputs, 0)
                encoder_outputs = torch.div(encoder_outputs, each_input_size)

                # =============================== 판별기 학습 =============================== #
                # 실제 팔레트를 이용한 BCE loss 계산.
                real = self.discriminator(real_palettes, encoder_outputs)
                d_loss_real = criterion_GAN(real, real_labels)

                # 가짜 팔레트를 이용한 BCE loss 계산.
                fake = self.discriminator(fake_palettes, encoder_outputs)
                d_loss_fake = criterion_GAN(fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake

                # Backprop and optimize.
                self.discriminator_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()

                # ================================ 생성기 학습 ================================= #
                # BCE loss 계산.
                fake = self.discriminator(fake_palettes, encoder_outputs)
                g_loss_GAN = criterion_GAN(fake, real_labels)

                # smooth L1 loss 계산.
                g_loss_smoothL1 = criterion_smoothL1(fake_palettes, real_palettes)

                # KL loss 계산.
                kl_loss = KL_loss(mu, logvar)

                g_loss = g_loss_GAN + g_loss_smoothL1 * self.args.lambda_sL1 + kl_loss * self.args.lambda_KL

                # Backprop and optimize.
                self.decoder_optimizer.zero_grad()
                g_loss.backward()
                self.decoder_optimizer.step()

            # For debugging. Save training output.
            if (epoch + 1) % self.args.sample_interval == 0:
                for x in range(5):  # saving 5 samples
                    fig1, axs1 = plt.subplots(nrows=1, ncols=5)
                    input_text = ''
                    for idx in txt_embeddings[x]:
                        if idx.item() == 0: break
                        if self.args.language == 'kor_ft' or 'kor_gl':
                            input_text += self.input_dict.index2word[idx.item()] + ' '
                        else:
                            input_text += self.input_dict.index2word[idx.item()] + ' '
                    axs1[0].set_title(input_text)
                    for k in range(5):
                        lab = np.array([fake_palettes.data[x][3 * k],
                                        fake_palettes.data[x][3 * k + 1],
                                        fake_palettes.data[x][3 * k + 2]], dtype='float64')
                        rgb = lab2rgb_1d(lab)
                        axs1[k].imshow([[rgb]])
                        axs1[k].axis('off')
                    if self.args.language == 'kor_ft' or 'kor_gl':
                        fig1.savefig(os.path.join(self.args.train_sample_dir_kor,
                                              'epoch{}_sample{}.jpg'.format(epoch + 1, x + 1)))
                    else:
                        fig1.savefig(os.path.join(self.args.train_sample_dir_eng,
                                                  'epoch{}_sample{}.jpg'.format(epoch + 1, x + 1)))
                    plt.close()
                print('학습 샘플을 저장합니다...')

            if (epoch + 1) % self.args.log_interval == 0:
                elapsed_time = time.time() - start_time
                print('소요 시간 [{:.4f}], 에폭 [{}/{}], 판별기_loss: {:.6f}, 생성기_loss: {:.6f}, KL_Loss: {:.6f}'
                      .format(elapsed_time, (epoch + 1),self.args.num_epochs, d_loss.item(), g_loss.item(), kl_loss))

            if (epoch + 1) % self.args.save_interval == 0:
                if self.args.language == 'kor_ft':
                    torch.save(self.encoder.state_dict(),
                               os.path.join(self.args.t2p_dir, '{}_G_encoder_kor_ft.ckpt'.format(epoch + 1)))
                    torch.save(self.decoder.state_dict(),
                               os.path.join(self.args.t2p_dir, '{}_G_decoder_kor_ft.ckpt'.format(epoch + 1)))
                    torch.save(self.discriminator.state_dict(),
                               os.path.join(self.args.t2p_dir, '{}_D_kor_ft.ckpt'.format(epoch + 1)))
                elif self.args.language == 'kor_gl':
                    torch.save(self.encoder.state_dict(),
                               os.path.join(self.args.t2p_dir, '{}_G_encoder_kor_gl.ckpt'.format(epoch + 1)))
                    torch.save(self.decoder.state_dict(),
                               os.path.join(self.args.t2p_dir, '{}_G_decoder_kor_gl.ckpt'.format(epoch + 1)))
                    torch.save(self.discriminator.state_dict(),
                               os.path.join(self.args.t2p_dir, '{}_D_kor_gl.ckpt'.format(epoch + 1)))

                else:
                    torch.save(self.encoder.state_dict(),
                               os.path.join(self.args.t2p_dir, '{}_G_encoder.ckpt'.format(epoch + 1)))
                    torch.save(self.decoder.state_dict(),
                               os.path.join(self.args.t2p_dir, '{}_G_decoder.ckpt'.format(epoch + 1)))
                    torch.save(self.discriminator.state_dict(),
                               os.path.join(self.args.t2p_dir, '{}_D.ckpt'.format(epoch + 1)))

                print('모델 체크 포인트를 저장합니다...')

    def test_t2p(self):

        # matplotlib 한글 폰트 설정
        if self.args.language == 'kor_ft' or 'kor_gl':
            font_fname = 'C:/Windows/Fonts/MALGUNSL.ttf'
            font_family = font_manager.FontProperties(fname=font_fname).get_name()
            plt.rcParams["font.family"] = font_family

        # ab 로그 제작
        lab_log_fake = []
        lab_log_real = []

        # 모델 호출
        if self.args.model_epoch:
            self.load_model(self.args.mode, self.args.model_epoch)

        print('T2P 모델 테스트를 시작합니다...')
        for batch_idx, (txt_embeddings, real_palettes, _) in enumerate(self.test_loader):
            # # 탈출
            # if batch_idx == 1:
            #     break

            if txt_embeddings.size(0) != self.args.batch_size:
                break

            # Compute text input size (without zero padding).
            batch_size = txt_embeddings.size(0)
            nonzero_indices = list(torch.nonzero(txt_embeddings)[:, 0])
            # 각 인풋별 단어 수 계산 ex '공산주의' = 1 '어, 두운, 네온' = 3
            # text_embeddings = 인풋 내 단어별 임베딩 id 들고 있음 / each_input_size = 인풋 내 단어 수 들고 있음
            each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]

            # Prepare test data.
            txt_embeddings = txt_embeddings.to(self.device)
            real_palettes = real_palettes.to(self.device).float()

            # Generate multiple palettes from same text input.
            for num_gen in range(1): # 이미지당 만드는 개수 -> 10 에서 1로 수정

                # Prepare input and output variables.
                palette = torch.FloatTensor(batch_size, 3).zero_().to(self.device)
                fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(self.device)

                # ============================== Text-to-Palette ==============================#
                # Condition for the generator.
                encoder_hidden = self.encoder.init_hidden(batch_size).to(self.device)
                encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings, encoder_hidden)

                # 그림 저장 시 창으로 열리는것 방지
                matplotlib.use('Agg')

                # Generate color palette.
                for i in range(5):
                    # ================================ 어텐션 ================================#
                    # attention_weight = torch.FloatTensor()
                    #
                    palette, decoder_context, decoder_hidden, attention_weight = \
                        self.decoder_T2P(palette, decoder_hidden.squeeze(0), encoder_outputs, each_input_size, i)
                    fake_palettes[:, 3 * i:3 * (i + 1)] = palette

                    # ================================ 어텐션 ================================#
                    # attention_weight = attention_weight.squeeze(1).to(device='cpu')
                    # attention_weight = attention_weight.detach().numpy()
                    #
                    # attention_weight_tmp = attention_weight[:, :]
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111)
                    # cax = ax.matshow(attention_weight_tmp, cmap='bone')
                    # fig.colorbar(cax)
                    #
                    # # 축 설정
                    # ax.set_yticklabels(self.input_dict.index2word, rotation=90)
                    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                    # plt.show()

                    # ================================ Save Results ================================#

                for x in range(self.args.batch_size):
                    # # 탈출
                    # if x == 10 :
                    #     break

                    # Input text.
                    input_text = ''
                    for idx in txt_embeddings[x]:
                        if idx.item() == 0: break
                        if self.args.language == 'kor_ft' or 'kor_gl':
                            # 형태소 분석하면 띄어쓰기 넣어줘야 함 // 아닌 경우 + ' ' 삭제
                            input_text += self.input_dict.index2word[idx.item()] + ' '
                        else:
                            input_text += self.input_dict.index2word[idx.item()] + ' '



                    # Save palette generation results.
                    if self.args.rand == 0:
                        fig1, axs1 = plt.subplots(nrows=2, ncols=5)
                        axs1[0][0].set_title(input_text + ' fake {}'.format(num_gen + 1))
                        # ab 로그 제작
                        log_fake_tmp = [[], [], [], [], []]
                        log_fake_name = [[], [], [], [], [], [], []]
                        log_real_tmp = [[], [], [], [], []]

                        for k in range(5):
                            lab = np.array([fake_palettes.data[x][3 * k],
                                            fake_palettes.data[x][3 * k + 1],
                                            fake_palettes.data[x][3 * k + 2]], dtype='float64')
                            # ab 로그 제작
                            log_fake_tmp[k] = [lab[0], lab[1], lab[2]]
                            rgb = lab2rgb_1d(lab)
                            axs1[0][k].imshow([[rgb]])
                            axs1[0][k].axis('off')
                        # 이미지 변환
                        change_color_batch_id = self.args.batch_size * batch_idx + x + 1
                        # change_color(change_color_batch_id, log_fake_tmp, input_text)
                        change_color_hsv(change_color_batch_id, log_fake_tmp, input_text)
                        change_color_lch(change_color_batch_id, log_fake_tmp, input_text)

                        log_fake_name = [[change_color_batch_id], [input_text], log_fake_tmp[0], log_fake_tmp[1],
                                         log_fake_tmp[2], log_fake_tmp[3], log_fake_tmp[4]]
                        lab_log_fake.append(log_fake_name)

                        axs1[1][0].set_title(input_text + ' real')
                        for k in range(5):
                            lab = np.array([real_palettes.data[x][3 * k],
                                            real_palettes.data[x][3 * k + 1],
                                            real_palettes.data[x][3 * k + 2]], dtype='float64')
                            # ab 로그 제작
                            log_real_tmp[k] = [lab[0], lab[1], lab[2]]

                            rgb = lab2rgb_1d(lab)
                            axs1[1][k].imshow([[rgb]])
                            axs1[1][k].axis('off')
                        lab_log_real.append(log_real_tmp)
                    else:
                        # 설문용 상하 위치 랜덤
                        rand_num = randint(0, 1)
                        rand_num_ex = 1 if rand_num == 0 else 0
                        fig1, axs1 = plt.subplots(nrows=2, ncols=5)
                        axs1[0][0].set_title(input_text + '\n' + '(1)') # + ' fake {}'.format(num_gen + 1))
                        for k in range(5):
                            lab = np.array([fake_palettes.data[x][3 * k],
                                            fake_palettes.data[x][3 * k + 1],
                                            fake_palettes.data[x][3 * k + 2]], dtype='float64')
                            rgb = lab2rgb_1d(lab)
                            axs1[rand_num][k].imshow([[rgb]])
                            axs1[rand_num][k].axis('off')
                        axs1[1][0].set_title('(2)')
                        for k in range(5):
                            lab = np.array([real_palettes.data[x][3 * k],
                                            real_palettes.data[x][3 * k + 1],
                                            real_palettes.data[x][3 * k + 2]], dtype='float64')
                            rgb = lab2rgb_1d(lab)
                            axs1[rand_num_ex][k].imshow([[rgb]])
                            axs1[rand_num_ex][k].axis('off')

                    if self.args.language == 'kor_ft' or 'kor_gl':
                        fig1.savefig(os.path.join(self.args.test_sample_dir_kor, self.args.mode,
                                              '{}_palette{}.jpg'.format(self.args.batch_size * batch_idx + x + 1,
                                                                        num_gen + 1)))
                    else:
                        fig1.savefig(os.path.join(self.args.test_sample_dir_eng, self.args.mode,
                                              '{}_palette{}.jpg'.format(self.args.batch_size * batch_idx + x + 1,
                                                                        num_gen + 1)))
                    print('테스트 [{}], 문단 [{}]에 대한 결과 [{}]을/를 저장했습니다.'.format(
                        self.args.batch_size * batch_idx + x + 1, input_text, num_gen + 1))

        # ab 로그 파일 저장
        log_time = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
        log_path_fake = self.args.lab_dir + log_time + "_fake_lab_" + self.args.language + ".pkl"
        with open(log_path_fake, "wb") as pkl:
            pickle.dump(lab_log_fake, pkl)
        log_path_real = self.args.lab_dir + log_time + "_real_lab_" + self.args.language + ".pkl"
        with open(log_path_real, "wb") as pkl:
            pickle.dump(lab_log_real, pkl)
        # palette_diversity(log_path_fake, log_path_real)


    def train_p2c(self):
        # Loss function.
        criterion_GAN = nn.BCELoss()
        criterion_smoothL1 = nn.SmoothL1Loss()

        start_epoch = 0
        if self.args.resume_epoch:
            print(f'모델을 불러오고 {format(self.args.resume_epoch)} 에폭부터 학습을 시작합니다...')
            start_epoch = self.args.resume_epoch
            self.load_model(self.args.mode, self.args.resume_epoch)
        else:
            print('신규 학습을 시작합니다...')

        self.decoder.train()
        self.discriminator.train()

        # 시작 시간 저장
        start_time = time.time()
        for epoch in range(start_epoch, self.args.num_epochs):
            for i, (images, palettes) in enumerate(self.train_loader):
                # 트레이닝 데이터 준비
                palettes = palettes.view(-1, 5, 3).cpu().data.numpy()
                inputs, real_images, global_hint = self.prepare_data(images, palettes, self.args.always_give_global_hint,
                                                                     self.args.add_L)
                batch_size = inputs.size(0)

                # BCE 손실 계산을 위한 라벨 준비
                real_labels = torch.ones(batch_size).to(self.device)
                fake_labels = torch.zeros(batch_size).to(self.device)

                # =============================== 판별기 학습 =============================== #
                # 진짜 이미지 BCE 손실 계산
                real = self.discriminator(torch.cat((real_images, global_hint), dim=1))
                real = real.squeeze() # 차원이 안맞아 추가함
                d_loss_real = criterion_GAN(real, real_labels)

                # 가짜 이미지 BCE 손실 계산
                fake_images = self.decoder(inputs, global_hint) # global_hint
                fake = self.discriminator(torch.cat((fake_images, global_hint), dim=1))
                # fake = fake.squeeze()
                d_loss_fake = criterion_GAN(fake, fake_labels)

                d_loss = (d_loss_real + d_loss_fake) * self.args.lambda_GAN

                # Backprop and optimize.
                self.discriminator_optimizer.zero_grad()
                d_loss.backward()
                self.discriminator_optimizer.step()

                # ================================ 디코더 학습 ================================= #
                # Compute BCE loss (fool the discriminator).
                fake_images = self.decoder(inputs, global_hint)
                fake = self.discriminator(torch.cat((fake_images, global_hint), dim=1))
                g_loss_GAN = criterion_GAN(fake, real_labels)

                # Compute smooth L1 loss.
                outputs = fake_images.view(batch_size, -1)
                labels = real_images.contiguous().view(batch_size, -1)
                g_loss_smoothL1 = criterion_smoothL1(outputs, labels)

                g_loss = g_loss_GAN * self.args.lambda_GAN + g_loss_smoothL1

                # Backprop and optimize.
                self.decoder_optimizer.zero_grad()
                g_loss.backward()
                self.decoder_optimizer.step()

            if (epoch + 1) % self.args.log_interval == 0:
                elapsed_time = time.time() - start_time
                print('소요 시간 [{:.4f}], 에폭 [{}/{}], 판별기_loss: {:.6f}, 생성기_loss: {:.6f}'.format(
                    elapsed_time, (epoch + 1), self.args.num_epochs, d_loss.item(), g_loss.item()))

            if (epoch + 1) % self.args.save_interval == 0:
                torch.save(self.decoder.state_dict(),
                           os.path.join(self.args.p2c_dir, '{}_G.ckpt'.format(epoch + 1)))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(self.args.p2c_dir, '{}_D.ckpt'.format(epoch + 1)))
                print('모델 체크포인트를 저장합니다...')