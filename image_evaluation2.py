import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import time


def color_histogram(file_path, result_path, name):
    img = cv2.imread(file_path + name)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
        plt.ylim([0, 120000])
    fig = plt.gcf()
    fig.savefig(result_path+name+'_hist.png', dpi=fig.dpi)

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

def image_color_cluster(file_path, result_path, name, k):
    image = cv2.imread(file_path + name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters=k)
    clt.fit(image)

    hist = centroid_histogram(clt)
    rgb = clt.cluster_centers_
    rgb = np.around(rgb, 0)
    percent = np.around(hist, 3)
    bar = plot_colors(hist, clt.cluster_centers_)

    plt.figure()
    # plt.axis("off")
    percent.sort()
    plt.title(percent)
    plt.xlabel(rgb)
    plt.imshow(bar)
    fig = plt.gcf()
    fig.savefig(result_path + name + '_cluster.png', dpi=fig.dpi)

file_path = './samples/color_hist/hist_files/'
result_path = './samples/color_hist/result/'
file_list = os.listdir(file_path)
for _, name in enumerate(file_list):
    start_time = time.time()
    color_histogram(file_path, result_path, name)
    elapsed_time = time.time() - start_time
    print(f'{name} color histogram 생성 완료[소요 시간: {elapsed_time // 60}분 {elapsed_time % 60}초]')
    plt.clf()
    image_color_cluster(file_path, result_path, name, k=5)
    elapsed_time = time.time() - start_time
    plt.clf()
    print(f'{name} 대표색 생성 완료[소요 시간: {elapsed_time // 60}분 {elapsed_time % 60}초]')
