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

        # Подготовка каждого канала
        b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
        b = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
        b = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

        g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
        g = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
        g = cv2.erode(g, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

        r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
        r = cv2.dilate(r, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
        r = cv2.erode(r, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

        # объединение 3х каналов в один
        temp = cv2.bitwise_and(b, b, mask=g)
        temp = cv2.bitwise_and(temp, temp, mask=r)
        #cv2.imshow("temp", temp)
        #№cv2.waitKey()

        rows = temp.shape[0]
        mask = np.zeros((height, width, 1), np.uint8)

        # поиск кругов (камня)
        #circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, rows, param1=150, param2=40, minRadius=50, maxRadius=350)
        circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, rows, param1=120, param2=20, minRadius=50,maxRadius=350)

        #for i in circles[0, :]:
        #    # draw the outer circle
        #    cv2.circle(image, (i[0], i[1]), int(i[2]), (0, 255, 0), 2)
        #    # draw the center of the circle
        #    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)


        # создание маски по найденному кругу
        circles = np.uint16(np.around(circles))
        i = circles[0, 0]
        center = (i[0], i[1])
        radius = i[2]

        #cv2.circle(mask, center, radius - 30, 255, cv2.FILLED, 8, 0)

        #gemAndMask = cv2.bitwise_and(image, image, mask=mask)
        #cv2.imshow("image", image)
        #cv2.waitKey()
        return center, radius





def temp():
    width = 1280
    height = 720
    groupp = 5
    name = "D:/GitHub/NIR/down2/5/5_8.jpg"

    image = cv2.imread(name)
    image = cv2.resize(image, (width, height))

    clas = HistClasification()
    image_clone = image
    center, radius = clas.function2(image_clone)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, center, radius - 30, 255, cv2.FILLED, 8, 0)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    fig, ax = plt.subplots()
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        print(i)

        if i == 0:
            histr = cv2.calcHist([hsv_image], [i], mask, [180], [0, 180])
        else:
            histr = cv2.calcHist([hsv_image], [i], mask, [256], [0, 256])

        #histr = cv2.calcHist([image], [i], mask, [256], [0, 256])

        ax.plot(histr, color=col, linewidth=2)

    print(1)

    ax.set_xlabel("интенсивность")
    ax.set_ylabel("количество пикселей")
    ax.grid()
    fig.set_figwidth(7)
    fig.set_figheight(5)
    plt.xlim([0, 256])
    plt.show()
    cv2.waitKey()

def temp3():
    width = 1280
    height = 720
    name = ("D:/GitHub/NIR/down2/1/1_2.jpg", "D:/GitHub/NIR/down2/2/2_2.jpg", "D:/GitHub/NIR/down2/3/3_2.jpg", "D:/GitHub/NIR/down2/4/4_2.jpg", "D:/GitHub/NIR/down2/5/5_2.jpg")

    clas = HistClasification()
    fig, ax = plt.subplots()
    color = ('b', 'g', 'r', "m", "k")

    for i in range(1):
        image = cv2.imread(name[i])
        image_clone = cv2.imread(name[i])
        center, radius = clas.function2(image_clone)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(mask, center, radius - 30, 255, cv2.FILLED, 8, 0)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        (H, S, V) = cv2.split(hsv_image)
        H = H * 2

        #histr = cv2.calcHist([image], [0], mask, [256], [0, 256])
        histr = cv2.calcHist([hsv_image], [0], mask, [180], [0, 180])

        ax.plot(histr, color=color[i], linewidth=3)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(20000))

    ax.grid(which='major', color='k')

    ax.minorticks_on()
    ax.grid(which='minor', color='gray', linestyle=':')

    ax.set_xlabel("интенсивность")
    ax.set_ylabel("количество пикселей")

    fig.set_figwidth(10)
    fig.set_figheight(8)
    plt.xlim([0, 180])
    plt.show()
    cv2.waitKey()


def temp2():
    clasificator2 = HistClasification()

    name = "D:/GitHub/NIR/down2/" + str(1) + "/" + str(1) + "_1.jpg"
    clasificator2.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down2/" + str(2) + "/" + str(2) + "_2.jpg"
    clasificator2.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down2/" + str(3) + "/" + str(3) + "_1.jpg"
    clasificator2.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down2/" + str(4) + "/" + str(4) + "_1.jpg"
    clasificator2.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down2/" + str(5) + "/" + str(5) + "_2.jpg"
    clasificator2.preparation_references(cv2.imread(name))

    #################
    """тестрование"""
    #################

    answer = []
    # количество папок\групп
    numGroups = len(glob.glob('D:/GitHub/NIR/down2/*'))
    for i in range(1, numGroups + 1):
        path = 'D:/GitHub/NIR/down2/' + str(i) + '/*'
        numGems = len(glob.glob(path))

        nameOne = "D:/GitHub/NIR/down2/" + str(i)
        nameTwo = ".jpg"

        for numImage in range(0, numGems):
            path_name = nameOne + "/" + str(i) + "_" + str(numImage) + nameTwo

            temp, likeness = clasificator2.group_definition(cv2.imread(path_name))
            if temp == i:
                answer.append(True)
            else:
                print("группа: ", i, "камень: ", numImage, "определен как: ", temp, "схожесть = ", likeness)

                answer.append(False)

    true_answer = 0
    false_answer = 0
    for i in range(len(answer)):
        if answer[i] == True:
            true_answer = true_answer + 1
        else:
            false_answer = false_answer + 1

    print("true", true_answer)
    print("false", false_answer)
    print("точность:", (true_answer / (true_answer + false_answer)))
    cv2.waitKey()




if __name__ == "__main__":
    temp()
    #temp3()
    #temp2()

    #################
    """подготовка"""
    #################


    clasificator = HistClasification()

    name = "D:/GitHub/NIR/down/" + str(1) + "/" + str(1) + "_1.jpg"
    clasificator.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down/" + str(2) + "/" + str(2) + "_2.jpg"
    clasificator.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down/" + str(3) + "/" + str(3) + "_1.jpg"
    clasificator.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down/" + str(4) + "/" + str(4) + "_1.jpg"
    clasificator.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down/" + str(5) + "/" + str(5) + "_3.jpg"
    clasificator.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down/" + str(6) + "/" + str(6) + "_1.jpg"
    clasificator.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down/" + str(7) + "/" + str(7) + "_1.jpg"
    clasificator.preparation_references(cv2.imread(name))

    name = "D:/GitHub/NIR/down/" + str(8) + "/" + str(8) + "_25.jpg"
    clasificator.preparation_references(cv2.imread(name))

    for i in range(9, 17):
        name = "D:/GitHub/NIR/down/" + str(i) + "/" + str(i) + "_1.jpg"
        clasificator.preparation_references(cv2.imread(name))
        #print(i)

    #################
    """тестрование"""
    #################

    answer = []
    # количество папок\групп
    numGroups = len(glob.glob('D:/GitHub/NIR/down/*'))
    for i in range(1, numGroups + 1):
        path = 'D:/GitHub/NIR/down/' + str(i) + '/*'
        numGems = len(glob.glob(path))

        nameOne = "D:/GitHub/NIR/down/" + str(i)
        nameTwo = ".jpg"

        for numImage in range(0, numGems):
            path_name = nameOne + "/" + str(i) + "_" + str(numImage) + nameTwo

            temp, likeness = clasificator.group_definition(cv2.imread(path_name))
            if temp == i:
                answer.append(True)
            else:
                print("группа: ", i, "камень: ", numImage)
                answer.append(False)

    true_answer = 0
    false_answer = 0
    for i in range(len(answer)):
        if answer[i] == True:
            true_answer = true_answer + 1
        else:
            false_answer = false_answer + 1

    print("true", true_answer)
    print("false", false_answer)


