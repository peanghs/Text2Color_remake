import torch
import pickle
import pandas as pd
import os
from konlpy.tag import Okt
from eunjeon import Mecab
import numpy as np
from skimage.color import deltaE_ciede2000



def make_excel():
    # path_ori = './data/hexcolor_vf/all_names.pkl'
    path_ori = './data/hexcolor_vf/test_names.pkl'
    # path_ori = './data/hexcolor_vf/train_names.pkl'
    with open(path_ori, 'rb') as f:
        text_data = pickle.load(f)
        print(len(text_data))
        pd_text_data = pd.DataFrame(text_data)
        # print(pd_text_data)
        # pd_text_data.to_excel(excel_writer='test_names_kor.xlsx')
        pd_text_data.to_csv('test_name.txt', index=False, header=None)
        print('----------done----------')
        f.close()

def morphs_convert():

    target = 'all'

# -------------------------------------

    target_name_txt = './data/text/' + target + '_names_kor.txt'
    target_name = './data/hexcolor_vf/kor_' + target + '_names.pkl'
    target_name_okt = './data/hexcolor_vf/kor_' + target + '_names_okt.pkl'
    target_name_mecab = './data/hexcolor_vf/kor_' + target + '_names_mecab.pkl'

    if target == 'all':
        target_palette_name = './data/hexcolor_vf/train_palettes_rgb.pkl'
    else:
        target_palette_name = './data/hexcolor_vf/' + target + '_palettes_rgb.pkl'

    okt = Okt()
    mecab = Mecab()
    name_seqs = []
    name_seqs_m = []

    # 텍스트 파일 전처리
    txt_file = open(target_name_txt, mode='rt', encoding='utf-8')
    file = txt_file.readlines()
    file = list(map(lambda s: s.strip(), file))
    print(f'----텍스트 라인 수 : {len(file)}----')

    # 일반 텍스트 피클 파일 생성
    with open(target_name,'wb') as pkl :
        pickle.dump(file, pkl)

    with open(target_name, "rb") as pkl:
        pkl_load = pickle.load(pkl)
        print(f'----일반 pkl 라인 수 : {len(pkl_load)}----')

    # # OKT 작업
    # for i, tmp in enumerate(file):
    #     tmp = okt.morphs(tmp)
    #     name_seqs.append(tmp)
    #
    # with open(target_name_okt,'wb') as pkl :
    #     pickle.dump(name_seqs, pkl)
    #
    # with open(target_name_okt, "rb") as pkl:
    #     pkl_load = pickle.load(pkl)
    #     print(f'----OKT pkl 라인 수 : {len(pkl_load)}----')
    #
    # # Mecab 작업
    # for j, tmp_m in enumerate(file):
    #     tmp_m = mecab.morphs(tmp_m)
    #     name_seqs_m.append(tmp_m)
    #
    # with open(target_name_mecab,'wb') as pkl :
    #     pickle.dump(name_seqs_m, pkl)
    #
    # with open(target_name_mecab, "rb") as pkl:
    #     pkl_load = pickle.load(pkl)
    #     print(f'----Mecab pkl 라인 수 : {len(pkl_load)}----')

    # 팔레트 라인과 비교
    with open(target_palette_name, 'rb') as f:
        palette_data = pickle.load(f)
        print(f'----팔레트 라인 수 : {len(palette_data)}----')


def palette_diversity(fake_lab, real_lab):

    # -------------------------------------
    # 한글 fast text
    # fake_lab = './log/T2P/09_20_22_08_fake_lab_kor_ft.pkl'
    # real_lab = './log/T2P/09_20_22_08_real_lab_kor_ft.pkl'

    # 영어
    # fake_lab = './log/T2P/09_20_23_44_fake_lab_eng.pkl'
    # real_lab = './log/T2P/09_20_23_44_real_lab_eng.pkl'

    # -------------------------------------

    with open(fake_lab, "rb") as pkl:
        pkl_load = pickle.load(pkl)
        fake_lab_np = np.array(pkl_load)
        print(f'-- fake 팔레트 수 : {len(pkl_load)} --')

    with open(real_lab, "rb") as pkl:
        pkl_load = pickle.load(pkl)
        real_lab_np = np.array(pkl_load)
        print(f'-- real 팔레트 수 : {len(pkl_load)} --')
    palette_len = int(len(fake_lab_np)/5)

    diff_np = np.zeros((palette_len, 4)) # 저장할 평균, 분산
    tmp = np.zeros((5,3)) # 팔레트 담는 임시
    # fake 계산
    for i, value in enumerate(fake_lab_np):
        z = i % 5
        tmp[z] = value
        diff_tmp = np.array([])  # 거리값 담는 임시
        if (i + 1) % 5 == 0:
            for j in range(4):
                for k in range(4-j):
                    diff_value = deltaE_ciede2000(tmp[j], tmp[k+j+1])
                    diff_tmp = np.append(diff_tmp, diff_value)
            diff_data = [np.mean(diff_tmp), np.std(diff_tmp)]
            palette_num = i // 5
            diff_np[palette_num][0] = diff_data[0]
            diff_np[palette_num][1] = diff_data[1]
            tmp = np.zeros((5, 3))
    print(f'-- fake 팔레트 {palette_num} 개 계산 완료 --')

    # real 계산
    for i, value in enumerate(real_lab_np):
        z = i % 5
        tmp[z] = value
        diff_tmp = np.array([])  # 거리값 담는 임시
        if (i + 1) % 5 == 0:
            for j in range(4):
                for k in range(4-j):
                    diff_value = deltaE_ciede2000(tmp[j], tmp[k+j+1])
                    diff_tmp = np.append(diff_tmp, diff_value)
            diff_data = [np.mean(diff_tmp), np.std(diff_tmp)]
            palette_num = i // 5
            diff_np[palette_num][2] = diff_data[0]
            diff_np[palette_num][3] = diff_data[1]
            tmp = np.zeros((5, 3))
    print(f'-- real 팔레트 {palette_num} 개 계산 완료 --')

    summary = np.mean(diff_np, axis=0)
    print(f'-- fake 팔레트 거리 평균 {summary[0]} --')
    print(f'-- fake 팔레트 분산 평균 {summary[1]} --')
    print(f'-- real 팔레트 거리 평균 {summary[2]} --')
    print(f'-- real 팔레트 분산 평균 {summary[3]} --')


# palette_diversity()
# morphs_convert()