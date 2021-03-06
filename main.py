
import cv2
import numpy as np
import math
import pandas as pd
from matplotlib import gridspec


def maskCreation2(inputImage, numberParts, gemGrupp):
    # разбиение изображения на каналы
    (bbb,ggg,rrr) = cv2.split(inputImage)

    # Подготовка каждого канала
    bbb = cv2.adaptiveThreshold(bbb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
    bbb = cv2.dilate(bbb, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    bbb = cv2.erode(bbb, cv2.getStructuringElement(cv2.MORPH_RECT, (1,1)))

    ggg = cv2.adaptiveThreshold(ggg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
    ggg = cv2.dilate(ggg, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    ggg = cv2.erode(ggg, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

    rrr = cv2.adaptiveThreshold(rrr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 7)
    rrr = cv2.dilate(rrr, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    rrr = cv2.erode(rrr, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

    # объединение 3х каналов в один
    temp = cv2.bitwise_and(bbb, bbb, mask=ggg)
    temp = cv2.bitwise_and(temp, temp, mask=rrr)

    rows = temp.shape[0]
    mask = np.zeros((1080, 1920, 1), np.uint8)
    # поиск кругов (камня)
    circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, rows,
                              param1=150, param2=40,
                              minRadius=200, maxRadius=500)
    # создание маски по найденному кругу
    circles = np.uint16(np.around(circles))
    i = circles[0, 0]
    center = (i[0], i[1])
    radius = i[2]
    cv2.circle(mask, center, radius - 30, 255, cv2.FILLED, 8, 0)

    cv2.imshow("gem", mask)
    cv2.waitKey()

    gemAndMask = cv2.bitwise_and(inputImage, inputImage, mask=mask)

    cv2.imshow("gem", gemAndMask)
    cv2.waitKey()

    # создание палитры
    palet = paletteCreation(gemAndMask, numberParts, i[1], i[0], radius)
    # создание вектора
    vector = vectorCreation(palet, numberParts, radius, gemGrupp)
    return vector



"""
Создание вектора из палитры 2х уровней
Параметры:  palette - палитра готовых цветов (круг с основными цветами)
            numberParts - количество кластеров
            centerX, centerY, radius - координаты центра палитры и радиус
            gemGrupp - номер группы камней
"""
def vectorCreation2(palette, numberParts, radius, gemGrupp):
    # cv2.imshow("palet", palette)
    # cv2.waitKey()

    paletHSV = cv2.cvtColor(palette, cv2.COLOR_BGR2HSV)
    alpha = int(360 / numberParts)
    startAngle = alpha/2 + 2
    endAngle = alpha + startAngle

    centerY = int(1920/ 2)
    centerX = int(1080 / 2)

    r1 = 150
    r2 = 350
    vector = []

    vector.append('{0}'.format(gemGrupp))

    for i in range(numberParts):
        x = int(centerX + r1 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
        y = int(centerY + r1 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))
        cv2.circle(palette, (centerY, centerX), 15, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(palette, (y,x), 15, (255,255,255), 3, cv2.LINE_AA)
        h, s, v = np.uint8(paletHSV[x, y])

        colorCode = 1000000 * h + 1000 * s + v
        vector.append('{0}'.format(colorCode))
        startAngle = startAngle + alpha
        endAngle = endAngle + alpha

    for i in range(numberParts):
        x = int(centerX + r2 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
        y = int(centerY + r2 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))
        cv2.circle(palette, (centerY, centerX), 15, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(palette, (y,x), 15, (255,255,255), 3, cv2.LINE_AA)
        h, s, v = np.uint8(paletHSV[x, y])
        colorCode = 1000000 * h + 1000 * s + v
        vector.append('{0}'.format(colorCode))
        startAngle = startAngle + alpha
        endAngle = endAngle + alpha

    #cv2.imshow("palet", palette)
    #cv2.waitKey()
    return vector






"""
Формирование палитры из подготовленного входгного изображения
Параметры:  inputImage - камень с маской
            numberParts - количесто кластеров
            centerX, centerY, radius - центр камня и его размер
"""
def paletteCreation2(inputImage, numberParts, centerX, centerY, radius):
    print("создание палитры")
    palette = np.zeros((1080, 1920, 3), np.uint8)

    alpha = int(360 / numberParts)
    startAngle = 0
    endAngle = alpha

    center = [int(1920 / 2), int(1080 / 2)]
    size = [400, 400]
    angle = 0

    x = 0
    y = 0
    r1 = int(radius / 4)
    r2 = int(radius - radius / 6)
    newR1 = r1
    newR2 = r2
    axes1 = (400, 400)
    axes2 = (200, 200)


    for i in range(numberParts):
        x = int(centerX + newR1 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
        y = int(centerY + newR1 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))

        b, g, r = np.uint8(inputImage[x, y])

        cv2.ellipse(palette, (center[0], center[1]), axes1, 0, int(startAngle), int(endAngle), (int(b), int(g), int(r)),-1)
        startAngle = startAngle + alpha
        endAngle = endAngle + alpha

    #cv2.imshow("palet1", palette)
    #cv2.waitKey()

    for i in range(numberParts):
        x = int(centerX + newR2 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
        y = int(centerY + newR2 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))

        b, g, r = np.uint8(inputImage[x, y])
        cv2.ellipse(palette, (center[0], center[1]), axes2, 0, int(startAngle), int(endAngle), (int(b), int(g), int(r)), -1)
        startAngle = startAngle + alpha
        endAngle = endAngle + alpha

    cv2.imshow("palet1", palette)
    cv2.waitKey()

    return palette






def paletteCreation(inputImage, numberParts, centerX, centerY, radius):
    print("создание палитры")
    palette = np.zeros((1080, 1920, 3), np.uint8)

    alpha = int(360 / numberParts)
    startAngle = 0
    endAngle = alpha

    center = [int(1920 / 2), int(1080 / 2)]
    size = [400, 400]
    angle = 0
    
    x = 0
    y = 0
    r = int(radius / 4)
    newR = r
    axes = (400, 400)

    for i in range(numberParts):
        x = int(centerX + newR * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
        y = int(centerY + newR * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))

        b,g,r = np.uint8(inputImage[x,y])
        cv2.ellipse(palette, (center[0], center[1]), axes, 0, int(startAngle), int(endAngle), (int(b), int(g), int(r)), -1)
        startAngle = startAngle + alpha
        endAngle = endAngle + alpha
    cv2.imshow("palet1", palette)
    cv2.waitKey()
    return palette


"""
Создание вектора из палитры
Параметры:  palette - палитра готовых цветов (круг с основными цветами)
            numberParts - количество кластеров
            centerX, centerY, radius - координаты центра палитры и радиус
            gemGrupp - номер группы камней
"""
def vectorCreation(palette, numberParts, radius, gemGrupp):
    # cv2.imshow("palet", palette)
    # cv2.waitKey()

    paletHSV = cv2.cvtColor(palette, cv2.COLOR_BGR2HSV)
    alpha = int(360 / numberParts)
    startAngle = alpha/2 + 2
    endAngle = alpha + startAngle

    centerY = int(1920/ 2)
    centerX = int(1080 / 2)

    r = 150
    vector = []

    vector.append('{0}'.format(gemGrupp))

    for i in range(numberParts):
        x = int(centerX + r * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
        y = int(centerY + r * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))
        cv2.circle(palette, (centerY, centerX), 15, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(palette, (y,x), 15, (255,255,255), 3, cv2.LINE_AA)
        h, s, v = np.uint8(paletHSV[x, y])
        print(h)
        print(s)
        print(v)
        colorCode = 1000000 * h + 1000 * s + v
        vector.append('{0}'.format(colorCode))
        startAngle = startAngle + alpha
        endAngle = endAngle + alpha

    #cv2.imshow("palet", palette)
    #cv2.waitKey()
    #print("vector")
    #print(vector)
    return vector

""" 
Данная функция принимает на вход изображения относящиеся к ондной группе
Параметры:  nameImage - массив имен входных изображений
            numberParts - количество кластеров цвета
            gemGrupp - номер группы
"""
def gruppCreation(nameImage, numberParts, gemGrupp):
    numberGem = len(nameImage)  # Определение количества входных имен
    print("количество имен ")
    print(numberGem)
    row = 0
    columns = []

    # подготовка таблицы
    for j in range(0, numberParts+1):
        columns.append('{0}'.format(j))

    array = pd.DataFrame(columns=columns)

    for i in range(numberGem):
        print("Шаг")
        print(i)
        gem = cv2.imread(nameImage[i])
        vector = maskCreation2(gem, numberParts, gemGrupp)

        maxInd = numberParts + 1
        array.loc[row] = vector
        row = row + 1
    return array


def gruppCreation2(nameImage, numberParts, gemGrupp):
    numberGem = len(nameImage)  # Определение количества входных имен
    row = 0
    columns = []

    # подготовка таблицы
    for j in range(0, (numberParts) + 1):
        columns.append('{0}'.format(j))

    array = pd.DataFrame(columns=columns)

    for i in range(numberGem):
        print("Шаг")
        print(i)
        gem = cv2.imread(nameImage[i])
        vector = maskCreation2(gem, numberParts, gemGrupp)

        maxInd = numberParts + 1
        array.loc[row] = vector
        row = row + 1
    return array





# Тесты для Алины

columns = []

    # подготовка таблицы
for j in range(0, 10):
    columns.append('{0}'.format(j))

array = pd.DataFrame(columns=columns)

testVecctor = [1, 2, 111, 3, 4, 667, 7, 4, 6, 10]

str0 = []
str0.append('{0}'.format(t))
str0.append('{0}'.format(x))

# str0 = (t, x)

for i in range(10):
    array.loc[i] = testVecctor

print(array)

array.to_csv("./testForAlina.csv", index=None, header=True)








"""





# otchet

name = []
name1 = "./inputImage/1_6.jpg"
name.append(name1)

ar = gruppCreation2(name, 4,1)


# main
klast = 4

#Группа 1 
nameOne = "./inputImage/1_"
nameTwo = ".jpg"

name = []
for i in range(1,10):
    temp = nameOne + str(i) + nameTwo
    name.append(temp)


array1 = gruppCreation2(name, klast, 1)
print("конец 1")
# array.to_csv("./grupp1_12.csv", index=None, header=True)

# Группа 2 
nameOne = "./inputImage/2_"
nameTwo = ".jpg"
name = []
for i in range(1,10):
    temp = nameOne + str(i) + nameTwo
    name.append(temp)

array2 = gruppCreation2(name, klast, 2)
print("конец 2")


# Группа 3 
nameOne = "./inputImage/3_"
nameTwo = ".jpg"
name = []
for i in range(1,10):
    temp = nameOne + str(i) + nameTwo
    name.append(temp)

array3 = gruppCreation2(name, klast, 3)
print("конец 3")



tab4 = pd.concat((array1, array2, array3), ignore_index=True)
print(tab4)

tab4.to_csv("./24_clast.csv", index=None, header=True)


"""
