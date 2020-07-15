import os
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

    # 모델 설정
    parser.add_argument('--mode', type=str, default='train_t2p',
                        choices=['train_t2p', 'train_PCN', 'test_t2p', 'test_text2colors'])
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')