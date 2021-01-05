import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import time
import shutil
from skimage.color import lab2rgb, rgb2lab, deltaE_ciede2000
import warnings
import matplotlib


def color_histogram(file_path, result_path, name, mod):
    ff = np.fromfile(file_path + name, np.uint8)
    img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
        plt.ylim([0, 120000])
    fig = plt.gcf()
    fig.savefig(result_path + name[0:-4] + mod + '_hist.png', dpi=fig.dpi)

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

def image_color_cluster(file_path, result_path, name, mod, k):
    # 한글 경로 읽는 문제로 imread 가 아닌 딴거 씀
    ff = np.fromfile(file_path + name, np.uint8)
    image = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters=k)
    clt.fit(image)

    hist = centroid_histogram(clt)
    rgb = clt.cluster_centers_
    # rgb = np.around(rgb, 0)
    percent = np.around(hist, 3)
    bar = plot_colors(hist, clt.cluster_centers_)

    plt.figure()
    # plt.axis("off")
    plt.title(percent)
    plt.xlabel(rgb)
    plt.imshow(bar)
    fig = plt.gcf()
    fig.savefig(result_path + name[0:-4] + mod + '_cluster.png', dpi=fig.dpi)
    return rgb, percent

def pallet_table(file_path):
    pal_lab = []
    pal_lab_tmp = []
    file_list = os.listdir(file_path)
    for _, name in enumerate(file_list):
        img_ori = plt.imread(file_path + name)
        img_lab = rgb2lab(img_ori)
        pallet_point = [123, 221, 320, 418, 541]
        pal_lab_tmp = []
        for i, pv in enumerate(pallet_point):
            pal_lab_tmp = pal_lab_tmp + [[img_lab[147][pv][0], img_lab[147][pv][1], img_lab[147][pv][2]]]
        pal_lab.append(pal_lab_tmp)
    return pal_lab

def palette_diversity(fake_lab, real_lab):

    fake_lab_np = np.array(fake_lab)
    real_lab_np = np.array(real_lab)
    palette_len = int(len(fake_lab_np))

    diff_np = np.zeros((palette_len, 2)) # 저장할 평균, 분산
    # fake 계산
    for i, value in enumerate(fake_lab_np): # i 이미지
        diff_tmp = np.zeros(5)
        for j in range(len(value)): # j 팔레트
            fake_pal_cal = value[j]
            diff_last = 999
            for k in range(len(real_lab_np[i])): # k real 팔레트
                real_pal_cal = real_lab_np[i][k]
                diff_value = deltaE_ciede2000(fake_pal_cal, real_pal_cal)
                if diff_value < diff_last:
                    diff_last = diff_value
            diff_tmp[j] = diff_last

        diff_data = [np.mean(diff_tmp), np.std(diff_tmp)]
        diff_np[i][0] = diff_data[0]
        diff_np[i][1] = diff_data[1]
    return diff_np

def cal_per(per):
    fake_per = np.array(per)
    ori_per = [0.124, 0.162, 0.168, 0.272, 0.275]
    palette_len = int(len(fake_per))
    diff_per = np.zeros((palette_len,1))
    for i, value in enumerate(fake_per): # i 이미지
        diff_per_tmp = np.zeros(5)
        for j in range(len(value)): # 팔레트 0~4
            diff_per_tmp[j] = abs(value[j] - ori_per[j])
        diff_per[i] = [np.mean(diff_per_tmp)]
    return diff_per


# 그림 저장 시 창으로 열리는것 방지
matplotlib.use('Agg')

pal_file_path = './samples/color_hist/pal_files/'

hsv_file_path = './samples/color_hist/files/hsv/'
hsv_result_path = './samples/color_hist/result/hsv/'

lab_file_path = './samples/color_hist/files/lab/'
lab_result_path = './samples/color_hist/result/lab/'

lch_file_path = './samples/color_hist/files/lch/'
lch_result_path = './samples/color_hist/result/lch/'

real_pal_lab = pallet_table(pal_file_path)

warnings.filterwarnings(action='ignore')

start_time = time.time()
# =============== HSV ===============
hsv_file_list = os.listdir(hsv_file_path)
mod = '_hsv'
# 이미지(i) / 팔레트 번호(0~4) / lab(0~3)
hsv_pal_lab = []
hsv_per = []
for i, name in enumerate(hsv_file_list):
    color_histogram(hsv_file_path, hsv_result_path, name, mod)
    plt.clf()
    rgb, percent = image_color_cluster(hsv_file_path, hsv_result_path, name, mod, k=5)
    percent.sort()
    percent_tmp = percent.tolist()
    hsv_per.append(percent_tmp)
    rgb = rgb / 255.0
    hsv_pal_lab_tmp = rgb2lab(rgb)
    hsv_pal_lab_tmp = hsv_pal_lab_tmp.tolist()
    hsv_pal_lab.append(hsv_pal_lab_tmp)
    plt.clf()
    shutil.copy(hsv_file_path + name, hsv_result_path)
    if i % 10 == 9:
        progress = (i+1) / len(hsv_file_list) * 100
        elapsed_time = time.time() - start_time
        print(f'[HSV] {i+1}/{len(hsv_file_list)}  {progress:.0f}% 진행 완료 '
              f'[소요 시간: {elapsed_time // 3600}시간 {elapsed_time % 3600 // 60}분 {elapsed_time % 60}초]')

diff_hsv = palette_diversity(hsv_pal_lab, real_pal_lab)
diff_per_hsv = cal_per(hsv_per)
print('=============== HSV 계산 완료 ===============')

# =============== Lab ===============
lab_file_list = os.listdir(lab_file_path)
mod = '_lab'
# 이미지(i) / 팔레트 번호(0~4) / lab(0~3)
lab_pal_lab = []
lab_per = []
for i, name in enumerate(lab_file_list):
    color_histogram(lab_file_path, lab_result_path, name, mod)
    plt.clf()
    rgb, percent = image_color_cluster(lab_file_path, lab_result_path, name, mod, k=5)
    percent.sort()
    percent_tmp = percent.tolist()
    lab_per.append(percent_tmp)
    rgb = rgb / 255.0
    lab_pal_lab_tmp = rgb2lab(rgb)
    lab_pal_lab_tmp = lab_pal_lab_tmp.tolist()
    lab_pal_lab.append(lab_pal_lab_tmp)
    plt.clf()
    shutil.copy(lab_file_path + name, lab_result_path)
    if i % 10 == 9:
        progress = (i+1) / len(lab_file_list) * 100
        elapsed_time = time.time() - start_time
        print(f'[Lab] {i+1}/{len(lab_file_list)}  {progress:.0f}% 진행 완료 '
              f'[소요 시간: {elapsed_time // 3600}시간 {elapsed_time % 3600 // 60}분 {elapsed_time % 60}초]')

diff_lab = palette_diversity(lab_pal_lab, real_pal_lab)
diff_per_lab = cal_per(lab_per)
print('=============== Lab 계산 완료 ===============')


# =============== LCH ===============
lch_file_list = os.listdir(lch_file_path)
mod = '_lch'
# 이미지(i) / 팔레트 번호(0~4) / lab(0~3)
lch_pal_lab = []
lch_per = []
for i, name in enumerate(lch_file_list):
    color_histogram(lch_file_path, lch_result_path, name, mod)
    plt.clf()
    rgb, percent = image_color_cluster(lch_file_path, lch_result_path, name, mod, k=5)
    percent.sort()
    percent_tmp = percent.tolist()
    lch_per.append(percent_tmp)
    rgb = rgb / 255.0
    lch_pal_lab_tmp = rgb2lab(rgb)
    lch_pal_lab_tmp = lch_pal_lab_tmp.tolist()
    lch_pal_lab.append(lch_pal_lab_tmp)
    plt.clf()
    shutil.copy(lch_file_path + name, lch_result_path)
    if i % 10 == 9:
        progress = (i+1) / len(lch_file_list) * 100
        elapsed_time = time.time() - start_time
        print(f'[LCH] {i+1}/{len(lch_file_list)}  {progress:.0f}% 진행 완료 '
              f'[소요 시간: {elapsed_time // 3600}시간 {elapsed_time % 3600 // 60}분 {elapsed_time % 60}초]')

diff_lch = palette_diversity(lch_pal_lab, real_pal_lab)
diff_per_lch = cal_per(lch_per)
print('=============== LCH 계산 완료 ===============')

# =============== Total ===============

total_diff_table = np.concatenate((diff_hsv, diff_lab, diff_lch), axis=1) # hsv 평균, 분산, lab 평균, 분산, lch 평균, 분산
hsv_av = np.mean(total_diff_table[:, 0])
lab_av = np.mean(total_diff_table[:, 2])
lch_av = np.mean(total_diff_table[:, 4])
total_diff_per_table = np.concatenate((diff_per_hsv, diff_per_lab, diff_per_lch), axis=1) # hsv 평균, lab 평균, lch 평균
total_col_diff = [hsv_av, lab_av, lch_av]
total_per_diff = [np.mean(diff_per_hsv), np.mean(diff_per_lab), np.mean(diff_per_lch)]
print(f'[컬러 팔레트 평균] hsv: {hsv_av:.3f}, lab: {lab_av:.3f}, lch: {lch_av:.3f}')
print(f'[퍼센트 차이 평균] hsv: {total_per_diff[0]:.3f}, lab: {total_per_diff[1]:.3f}, lch: {total_per_diff[2]:.3f}')
np.savetxt('./samples/color_hist/result/col_diff.txt', total_diff_table, header="write start", footer="write end",
           fmt="%.3f")
np.savetxt('./samples/color_hist/result/per_diff.txt', total_diff_per_table, header="write start", footer="write end",
           fmt="%.3f")
elapsed_time = time.time() - start_time
print(f'[총 소요 시간: {elapsed_time // 3600}시간 {elapsed_time % 3600 // 60}분 {elapsed_time % 60}초]')