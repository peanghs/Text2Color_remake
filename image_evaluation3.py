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

pal_file_path = './samples/color_hist/st_diff/pal_files/'

stf_file_path = './samples/color_hist/st_diff/files/stf/'
stf_result_path = './samples/color_hist/st_diff/result/stf/'

stl_file_path = './samples/color_hist/st_diff/files/stl/'
stl_result_path = './samples/color_hist/st_diff/result/stl/'

real_pal_lab = pallet_table(pal_file_path)

warnings.filterwarnings(action='ignore')

start_time = time.time()
# =============== 스타일 선적용 ===============
stf_file_list = os.listdir(stf_file_path)
mod = '_stf'
# 이미지(i) / 팔레트 번호(0~4) / lab(0~3)
stf_pal_lab = []
stf_per = []
for i, name in enumerate(stf_file_list):
    color_histogram(stf_file_path, stf_result_path, name, mod)
    plt.clf()
    rgb, percent = image_color_cluster(stf_file_path, stf_result_path, name, mod, k=5)
    percent.sort()
    percent_tmp = percent.tolist()
    stf_per.append(percent_tmp)
    rgb = rgb / 255.0
    stf_pal_lab_tmp = rgb2lab(rgb)
    stf_pal_lab_tmp = stf_pal_lab_tmp.tolist()
    stf_pal_lab.append(stf_pal_lab_tmp)
    plt.clf()
    shutil.copy(stf_file_path + name, stf_result_path)
    if i % 10 == 9:
        progress = (i+1) / len(stf_file_list) * 100
        elapsed_time = time.time() - start_time
        print(f'[스타일 선 적용] {i+1}/{len(stf_file_list)}  {progress:.0f}% 진행 완료 '
              f'[소요 시간: {elapsed_time // 3600:.0f}시간 {elapsed_time % 3600 // 60:.0f}분 {elapsed_time % 60:.0f}초]')

diff_stf = palette_diversity(stf_pal_lab, real_pal_lab)
diff_per_stf = cal_per(stf_per)
print('=============== 스타일 선 적용 계산 완료 ===============')

# =============== 스타일 후 적용 ===============
stl_file_list = os.listdir(stl_file_path)
mod = '_stl'
# 이미지(i) / 팔레트 번호(0~4) / lab(0~3)
stl_pal_lab = []
stl_per = []
for i, name in enumerate(stl_file_list):
    color_histogram(stl_file_path, stl_result_path, name, mod)
    plt.clf()
    rgb, percent = image_color_cluster(stl_file_path, stl_result_path, name, mod, k=5)
    percent.sort()
    percent_tmp = percent.tolist()
    stl_per.append(percent_tmp)
    rgb = rgb / 255.0
    stl_pal_lab_tmp = rgb2lab(rgb)
    stl_pal_lab_tmp = stl_pal_lab_tmp.tolist()
    stl_pal_lab.append(stl_pal_lab_tmp)
    plt.clf()
    shutil.copy(stl_file_path + name, stl_result_path)
    if i % 10 == 9:
        progress = (i+1) / len(stl_file_list) * 100
        elapsed_time = time.time() - start_time
        print(f'[스타일 후 적용] {i+1}/{len(stl_file_list)}  {progress:.0f}% 진행 완료 '
              f'[소요 시간: {elapsed_time // 3600:.0f}시간 {elapsed_time % 3600 // 60:.0f}분 {elapsed_time % 60:.0f}초]')

diff_stl = palette_diversity(stl_pal_lab, real_pal_lab)
diff_per_stl = cal_per(stl_per)
print('=============== 스타일 후 적용 계산 완료 ===============')

# =============== Total ===============

total_diff_table = np.concatenate((diff_stf, diff_stl), axis=1)
stf_av = np.mean(total_diff_table[:, 0])
stl_av = np.mean(total_diff_table[:, 2])
total_diff_per_table = np.concatenate((diff_per_stf, diff_per_stl), axis=1)
total_col_diff = [stf_av, stl_av]
total_per_diff = [np.mean(diff_per_stf), np.mean(diff_per_stl)]
print(f'[컬러 팔레트 평균] 선 적용: {stf_av:.3f}, 후 적용: {stl_av:.3f}')
print(f'[퍼센트 차이 평균] 선 적용: {total_per_diff[0]:.3f}, 후 적용: {total_per_diff[1]:.3f}')
np.savetxt('./samples/color_hist/st_diff/result/st_col_diff.txt', total_diff_table, header="write start",
           footer="write end", fmt="%.3f")
np.savetxt('./samples/color_hist/st_diff/result/st_per_diff.txt', total_diff_per_table, header="write start",
           footer="write end", fmt="%.3f")
elapsed_time = time.time() - start_time
print(f'[총 소요 시간: {elapsed_time // 3600:.0f}시간 {elapsed_time % 3600 // 60:.0f}분 {elapsed_time % 60:.0f}초]')