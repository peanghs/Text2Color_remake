import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
from process_main import main_solver


def main(args):

    # 폴더가 없으면 새로 만듬
    if not os.path.exists(args.t2p_dir):
        os.makedirs(args.t2p_dir)

    solver = main_solver(args)

    if args.mode == 'train_t2p':
        solver.train_t2p()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    os.chdir('C:/Users/peang/PycharmProjects/Text2Color_remake')  # 기본 경로 변경(상대경로 입력 시)
    # 공통
    parser.add_argument('--mode', type=str, default='train_t2p',
                        choices=['train_t2p', 'train_PCN', 'test_t2p', 'test_text2colors'])

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    # text2palette

    # 경로
    parser.add_argument('--t2p_dir', type=str, default='./models/TPN')


    args = parser.parse_args()
    print(args)
    main(args)
