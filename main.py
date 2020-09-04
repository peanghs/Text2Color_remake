import os
# dll 중복 로드 에러나는 경우
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
from process_main import main_solver


def main(args):
    # 폴더가 없으면 새로 만듬
    if not os.path.exists(args.t2p_dir):
        os.makedirs(args.t2p_dir)

    solver = main_solver(args)

    if args.mode == 'train_t2p':
        solver.train_t2p()

    if args.mode == 'test_t2p':
        solver.test_t2p()

    if args.mode == 'train_p2c':
        solver.train_p2c()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    os.chdir('D:/Pycharm Project/Text2Color_remake/')  # 기본 경로 변경(상대경로 입력 시)
    # 공통
    parser.add_argument('--mode', type=str, default='train_t2p',
                        choices=['train_t2p', 'train_p2c', 'test_t2p', 'test_text2colors'])
    parser.add_argument('--dataset', type=str, default='bird256', choices=['imagenet', 'bird256'])

    # 경로
    parser.add_argument('--t2p_dir', type=str, default='./models/T2P')
    parser.add_argument('--train_sample_dir', type=str, default='./samples/train')
    parser.add_argument('--test_sample_dir', type=str, default='./samples/test')
    parser.add_argument('--p2c_dir', type=str, default='./models/P2C')


    # 모델 설정
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs for training')
    parser.add_argument('--dropout_p', type=float, default=0.1) #0.2
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate') #5e-4

    # 학습 관련
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lambda_sL1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_KL', type=float, default=0.5, help='weight for KL loss')

    #저장 관련
    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many steps to wait before logging training status')
    parser.add_argument('--sample_interval', type=int, default=20,
                        help='how many steps to wait before saving the training output')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='how many steps to wait before saving the trained models')


    # 실행 관련
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this epoch')
    # 테스트 시 모델을 불러오려면 입력해줘야 함
    parser.add_argument('--model_epoch', type=int, default=None, help='입력된 에폭의 모델을 불러옵니다')

    # text2palette 개별 설정
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--n_layers', type=int, default=1)

    # palette2color 개별 설정
    parser.add_argument('--always_give_global_hint', type=int, default=1)
    parser.add_argument('--add_L', type=int, default=1)
    parser.add_argument('--lambda_GAN', type=float, default=0.1)


    args = parser.parse_args()
    print(args)
    main(args)
