import torch
import pickle
import pandas as pd


def make_excel():
    # path_ori = './data/hexcolor_vf/all_names.pkl'
    path_ori = './data/hexcolor_vf/test_names.pkl'
    # path_ori = './data/hexcolor_vf/train_names.pkl'
    with open(path_ori, 'rb') as f:
        text_data = pickle.load(f)
        print(len(text_data))
        pd_text_data = pd.DataFrame(text_data)
        # print(pd_text_data)
        pd_text_data.to_excel(excel_writer='test_names_kor.xlsx')
        print('----------done----------')
        f.close()

def convert_pkl():
    pallete_path = './data/hexcolor_vf/train_palettes_rgb.pkl'
    raw_file = open('./data/text/all_names_kor.txt', mode='rt', encoding='utf-8')
    file = raw_file.readlines()
    file = list(map(lambda s: s.strip(), file))
    print(file)
    with open("kor_all_names.pkl",'wb') as pkl :
        pickle.dump(file, pkl)
    with open("kor_all_names.pkl", "rb") as pkl:
        pkl_load = pickle.load(pkl)
        print(pkl_load)
        print(f'----pkl 라인 수 : {len(pkl_load)}----')
    print(f'----텍스트 라인 수 : {len(file)}----')
    with open(pallete_path, 'rb') as f:
        pallete_data = pickle.load(f)
        print(f'----팔레트 라인 수 : {len(pallete_data)}----')

def pallete_check():
    name_path = './data/hexcolor_vf/kor_train_names.pkl'
    pallete_path = './data/hexcolor_vf/train_palettes_rgb.pkl'
    with open(name_path, 'rb') as f:
        text_data = pickle.load(f)
        print(len(text_data))
    with open(pallete_path, 'rb') as f:
        pallete_data = pickle.load(f)
        print(len(pallete_data))

convert_pkl()
# pallete_check()
# make_excel()