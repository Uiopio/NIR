import cv2
import numpy as np
import math
import glob
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

class HistClasification:

    def __init__(self):

        self.gem_reference = [] # список эталонных камней
        self.gem_reference.append("start")

        self.hist_gem_reference = []  # список эталонных камней
        self.hist_gem_reference.append("start")

    # Подготовка эталонных камней. Падется одно изоюражение и вычисляется его гистограмма
    def preparation_references(self, image):
        # уменьшаем размер для ускорения работы
        width = 1280
        height = 720
        image = cv2.resize(image, (width, height))

        image_clone = image
        center, radius = self.function2(image_clone)
        #mask = np.zeros(image.shape[:2], np.uint8)
        mask = np.zeros((height, width, 1), np.uint8)
        cv2.circle(mask, center, radius - 30, 255, cv2.FILLED, 8, 0)


        #cv2.imshow("mask", mask)
        #cv2.imshow("image", image)
        #cv2.waitKey()

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, [180, 256, 256], [0, 180, 0, 256, 0, 256], accumulate=False)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        self.hist_gem_reference.append(hist)


    # фото камня -> группу
    def group_definition(self, image):
        # создаем маску с камнем
        width = 1280
        height = 720
        image = cv2.resize(image, (width, height))

        image_clone = image
        center, radius = self.function2(image_clone)


        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(mask, center, radius - 30, 255, cv2.FILLED, 8, 0)


        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, [180, 256, 256], [0, 180, 0, 256, 0, 256], accumulate=False)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # определеяем количество эталонных гитсограмм и сравниваем с каждой
        likeness = []
        method = 3
        for i in range(1, len(self.hist_gem_reference)):
            likeness.append(cv2.compareHist(hist, self.hist_gem_reference[i], method))

        max_likeness = likeness[0]
        groupp = 0
        for i in range(0, len(likeness)):
            if (method == 0) or (method == 2):
                if likeness[i] >= max_likeness:
                    max_likeness = likeness[i]
                    groupp = i + 1
            else:
                if likeness[i] <= max_likeness:
                    max_likeness = likeness[i]
                    groupp = i + 1



        return groupp, max_likeness

    def function2(self, image):
        width = 1280
        height = 720
        # Создает изображение с маской
        # разбиение изображения на каналы
        (b, g, r) = cv2.split(image)

        cv2.imwrite("D:/imageNir/1/1.jpg", image)
        cv2.imwrite("D:/imageNir/1/b_1.jpg", b)
        cv2.imwrite("D:/imageNir/1/g_1.jpg", g)
        cv2.imwrite("D:/imageNir/1/r_1.jpg", r)
        cv2.waitKey()

        # Подготовка каждого канала
        b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
        g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
        r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)

        cv2.imwrite("D:/imageNir/1/b_2.jpg", b)
        cv2.imwrite("D:/imageNir/1/g_2.jpg", g)
        cv2.imwrite("D:/imageNir/1/r_2.jpg", r)

        b = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        b = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

        g = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        g = cv2.erode(g, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

        r = cv2.dilate(r, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        r = cv2.erode(r, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

        cv2.imwrite("D:/imageNir/1/b_3.jpg", b)
        cv2.imwrite("D:/imageNir/1/g_3.jpg", g)
        cv2.imwrite("D:/imageNir/1/r_3.jpg", r)
        cv2.waitKey()


        # объединение 3х каналов в один
        temp = cv2.bitwise_and(b, b, mask=g)
        temp = cv2.bitwise_and(temp, temp, mask=r)


        cv2.imshow("temp", temp)
        cv2.imwrite("D:/imageNir/1/temp.jpg", temp)
        cv2.waitKey()

        rows = temp.shape[0]
        mask = np.zeros((height, width, 1), np.uint8)

        # поиск кругов (камня)
        #circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, rows, param1=150, param2=40, minRadius=50, maxRadius=350)
        circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, rows, param1=120, param2=20, minRadius=50,maxRadius=350)

        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), int(i[2]), (0, 255, 0), 10)
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)


        # создание маски по найденному кругу
        circles = np.uint16(np.around(circles))
        i = circles[0, 0]
        center = (i[0], i[1])
        radius = i[2]

        #cv2.circle(mask, center, radius - 30, 255, cv2.FILLED, 8, 0)

        #gemAndMask = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("image", image)
        cv2.imwrite("D:/imageNir/1/2.jpg", image)
        cv2.waitKey()
        return center, radius




if __name__ == "__main__":
    clasificator = HistClasification()

    #name = "D:/GitHub/NIR/down/" + str(1) + "/" + str(1) + "_1.jpg"
    #clasificator.preparation_references(cv2.imread(name))

    path_name = "D:/imageNir/down/6/6_22.jpg"
    temp, likeness = clasificator.group_definition(cv2.imread(path_name))