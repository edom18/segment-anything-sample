import cv2
import math
import matplotlib.pyplot as plt
from typing import List
from sklearn.cluster import KMeans
from collections import Counter

class ColorData:
    labels: List[str]
    rgbs: List[tuple]

    def __init__(self, labels, rgbs):

        assert len(labels) == len(rgbs), 'The number of labels and rgbs must be the same.'

        self.labels = labels
        self.rgbs = rgbs

    def getlabelAt(self, index):
        return self.labels[index]

def color_distance(rgb1, rgb2):
    return math.sqrt(sum((e1-e2)**2 for e1, e2 in zip(rgb1, rgb2)))

def closest_color(target_rgbs, rgb_list):
    result = [[color_distance(test, target) for test in rgb_list] for target in target_rgbs]
    result = [inner.index(min(inner)) for inner in result]
    most_common = Counter(result).most_common(1)[0][0]
    return most_common

def extract_colors(image_path, num_colors):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors)
    labels = kmeans.fit_predict(image)
    colors = kmeans.cluster_centers_

    return colors

def check(image_path: str, color_data: ColorData) -> int:
    colors = extract_colors(image_path, 6)
    # plot_colors(colors)
    # plt.show()

    converted_colors = [tuple(color) for color in colors]
    result_index = closest_color(converted_colors, color_data.rgbs)

    return result_index

def plot_colors(colors):
    plt.figure(figsize=(5, 2), dpi=80)
    plt.axis("off")
    plt.imshow([colors.astype(int)], aspect='auto')

def main():

    image_path = 'images/sample.jpg'

    color_data = ColorData(
        labels=['red', 'green', 'blue', 'white', 'black'],
        rgbs=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)],
    )

    result_index = check(image_path, color_data)

    result_label = color_data.getlabelAt(result_index)

    print(result_label)

if __name__ == '__main__':
    main()