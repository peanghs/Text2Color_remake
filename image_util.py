from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb, lab2lch, lch2lab
import matplotlib.pyplot as plt
import math
import os
import time
import pickle

def change_color(batch_id, pal_lab, pal_name):
    file_path = './samples/img/'
    result_path = './samples/result/'
    file_list = os.listdir(file_path)

    for _, name in enumerate(file_list):
        img_ori = plt.imread(file_path + name)
        img_lab = rgb2lab(img_ori, illuminant='D55')
        counter = [0, 0, 0, 0, 0, 0]
        for i in range(len(img_lab)):
            for j in range(len(img_lab[i])):
                point_ori = [img_lab[i][j][1], img_lab[i][j][2]]
                total_diff_last = 363  # lot((128+128)^2 + (128+128)^2) +1
                for k in range(5):
                    point_pal = [pal_lab[k][1], pal_lab[k][2]]
                    x_diff = point_ori[0] - point_pal[0]
                    y_diff = point_ori[1] - point_pal[1]
                    total_diff = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
                    if total_diff < total_diff_last:
                        point_ori_tmp = point_pal
                        total_diff_last = total_diff
                        counter_fin = k
                if img_lab[i][j][0] >= 95:  # 흰색 패스
                    counter[5] += 1
                # elif total_diff_last >= 80:
                #     counter[6] += 1
                else:
                    counter[counter_fin] += 1
                    img_lab[i][j][1], img_lab[i][j][2] = [point_ori_tmp[0], point_ori_tmp[1]]
        print(f'[{pal_name}의 {name[0:-4]} 적용 내역] 팔레트1:{counter[0]}회, 팔레트2:{counter[1]}회, 팔레트3:{counter[2]}회, '
              f'팔레트4:{counter[3]}회, 팔레트5:{counter[4]}회, 흰색 패스:{counter[5]}회')
        img_trans = lab2rgb(img_lab, illuminant='D55')
        plt.imsave(os.path.join(result_path, f'{batch_id}_{pal_name}_{name[0:-4]}.png'), img_trans)

def change_color_hsv(batch_id, pal_lab, pal_name):
    file_path = './samples/img/'
    result_path = './samples/result_hsv/'
    file_list = os.listdir(file_path)

    pal_hsv = lab2rgb(pal_lab, illuminant='D55')
    pal_hsv = rgb2hsv(pal_hsv)

    for _, name in enumerate(file_list):
        img_ori = plt.imread(file_path + name)
        img_hsv = rgb2hsv(img_ori)
        counter_hsv = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(img_hsv)):
            for j in range(len(img_hsv[i])):
                counter_180 = False
                point_ori = img_hsv[i][j][0]
                total_diff_last = 2  # 1 - 0 + 1
                for k in range(5):
                    point_pal = pal_hsv[k][0]
                    total_diff = abs(point_ori - point_pal)
                    # 180 도 이상 처리 예각으로 대체
                    if total_diff > 0.5:
                        total_diff = 1 - total_diff
                        counter_180 = True
                    if total_diff < total_diff_last:
                        point_ori_tmp = point_pal
                        total_diff_last = total_diff
                        counter_fin = k
                if img_hsv[i][j][1] <= 0.02 and img_hsv[i][j][2] >= 0.98 :  # 채도 3 이하 명도 98 이상 패스(흰색)
                    counter_hsv[5] += 1
                # elif total_diff_last >= 1.0: #1.41
                #     counter_hsv[6] += 1
                else:
                    counter_hsv[counter_fin] += 1
                    img_hsv[i][j][0] = point_ori_tmp
                if counter_180:
                    counter_hsv[6] += 1
        sum_total = counter_hsv[0]+counter_hsv[1]+counter_hsv[2]+counter_hsv[3]+counter_hsv[4]+counter_hsv[5]
        print(f'[{pal_name}의 {name[0:-4]} 적용 내역] 팔레트1:{counter_hsv[0]:,}회, 팔레트2:{counter_hsv[1]:,}회, '
              f'팔레트3:{counter_hsv[2]:,}회, 팔레트4:{counter_hsv[3]:,}회, 팔레트5:{counter_hsv[4]:,}회, '
              f'흰색 패스:{counter_hsv[5]:,}회, 팔레트 변환 :{sum_total-counter_hsv[5]:,}회'
              f'({sum_total-counter_hsv[5]-counter_hsv[6]:,}회), 총 변환 :{sum_total:,}회')
        img_trans = hsv2rgb(img_hsv)
        plt.imsave(os.path.join(result_path, f'{batch_id}_{pal_name}_{name[0:-4]}.png'), img_trans)

def change_color_lch(batch_id, pal_lab, pal_name):
    file_path = './samples/img/'
    result_path = './samples/result_lch/'
    file_list = os.listdir(file_path)
    pal_lch = lab2lch(pal_lab)

    for _, name in enumerate(file_list):
        img_ori = plt.imread(file_path + name)
        img_lab = rgb2lab(img_ori)
        img_lch = lab2lch(img_lab)
        counter_lch = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(img_lch)):
            for j in range(len(img_lch[i])):
                point_ori = img_lch[i][j][2]
                total_diff_last = 11  # lot((1-0)^2 + (1-0)^2) +1
                for k in range(5):
                    point_pal = pal_lch[k][2]
                    total_diff = abs(point_ori - point_pal)
                    # 180 도 이상 처리 예각으로 대체
                    if total_diff > 5:
                        total_diff = 10 - total_diff
                        counter_lch[6] += 1
                    if total_diff < total_diff_last:
                        point_ori_tmp = point_pal
                        total_diff_last = total_diff
                        counter_fin = k
                if img_lch[i][j][1] <= 0.2 and img_lch[i][j][0] >= 9.8 :  # 채도 3 이하 명도 98 이상 패스(흰색)
                    counter_lch[5] += 1
                # elif total_diff_last >= 1.2: #1.41
                #     counter_hsv[6] += 1
                else:
                    counter_lch[counter_fin] += 1
                    img_lch[i][j][2] = point_ori_tmp
        sum_total = counter_lch[0]+counter_lch[1]+counter_lch[2]+counter_lch[3]+counter_lch[4]+counter_lch[5]
        print(f'[{pal_name}의 {name[0:-4]} 적용 내역] 팔레트1:{counter_lch[0]:,}회, 팔레트2:{counter_lch[1]:,}회, '
              f'팔레트3:{counter_lch[2]:,}회, 팔레트4:{counter_lch[3]:,}회, 팔레트5:{counter_lch[4]:,}회, '
              f'흰색 패스:{counter_lch[5]:,}회, 팔레트 변환 :{sum_total-counter_lch[5]:,}회'
              f'({sum_total-counter_lch[5]-counter_lch[6]:,}회), 총 변환 :{sum_total:,}회')
        img_trans = lch2lab(img_lch)
        img_trans = lab2rgb(img_trans, illuminant='D55')
        plt.imsave(os.path.join(result_path, f'{batch_id}_{pal_name}_{name[0:-4]}.png'), img_trans)