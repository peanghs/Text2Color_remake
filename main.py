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