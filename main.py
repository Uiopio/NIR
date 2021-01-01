
import cv2
import numpy as np
import math
import pandas as pd
from matplotlib import gridspec


def maskCreation(inputImage, numberParts, gemGrupp):
    mask = cv2.inRange(inputImage, (140,140,140), (255,255,255))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    mask = cv2.bitwise_not(mask)


    one = [0, 0]
    rows, cols = mask.shape
    temp = 0
    flag = False

    for i in range(rows):
        for j in range(cols):
            temp = mask[i, j]
            if temp == 255:
                flag = True
                one[0] = i
                one[1] = j
                break

        if flag == True:
            break

    two = [0, 0]
    flag = False
    for i in range(cols):
        for j in range(rows):
            temp = mask[j, i]
            if temp == 255:
                flag = True
                two[0] = j
                two[1] = i
                break

        if flag == True:
            break

    three = [0, 0]
    flag = False
    i = 1
    for i in range(1, rows + 1):
        for j in range(cols):
            temp = mask[rows - i, j]
            if temp == 255:
                flag = True
                three[0] = rows - i
                three[1] = j
                break

        if flag == True:
            break

    four = [0, 0]
    flag = False
    i = 1
    for i in range(1, cols + 1):
        for j in range(rows):
            temp = mask[j, cols - i]
            if temp == 255:
                flag = True
                four[0] = j
                four[1] = cols - i
                break

        if flag == True:
            break

    print("точка 1: ", one[0], " ", one[1])
    print("точка 2: ", two[0], " ", two[1])
    print("точка 3: ", three[0], " ", three[1])
    print("точка 4: ", four[0], " ", four[1])

    x = (int)((one[1] + three[1]) / 2)
    y = (int)((two[0] + four[0]) / 2)

    print("центр: ", x, " ", y)
    radius = (int)(y - one[0] - 30)

    newMask = np.zeros((1080, 1920, 1), np.uint8)

    cv2.circle(newMask, (x,y), radius, 255, cv2.FILLED, 8, 0)
    gemAndMask = cv2.bitwise_and(inputImage, inputImage, mask=newMask)

    palet = paletteCreation(gemAndMask, numberParts, y, x, radius)

    vector = vectorCreation(palet, numberParts, y, x, radius, gemGrupp)
    return vector


"""
Формирование палитры из подготовленного входгного изображения
Параметры:  inputImage - камень с маской
            numberParts - количесто кластеров
            centerX, centerY, radius - центр камня и его размер
"""
def paletteCreation(inputImage, numberParts, centerX, centerY, radius):
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

    isBlack = True
    black = 0
    for i in range(numberParts):
        while isBlack == True:
            x = int(centerX + newR * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
            y = int(centerY + newR * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))

            b,g,r = np.uint8(inputImage[x,y])
            black = b + g + r

            if black == 0:
                isBlack = True
                newR = newR + 50
            else:
                isBlack = False
                black = 0
                newR = r

            if newR > r:
                newR = 10

        isBlack = True
        cv2.ellipse(palette, (center[0], center[1]), axes, 0, int(startAngle), int(endAngle), (int(b), int(g), int(r)), -1)
        startAngle = startAngle + alpha
        endAngle = endAngle + alpha

    return palette


"""
Создание вектора из палитры
Параметры:  palette - палитра готовых цветов (круг с основными цветами)
            numberParts - количество кластеров
            centerX, centerY, radius - координаты центра палитры и радиус
            gemGrupp - номер группы камней
"""
def vectorCreation(palette, numberParts, centerX, centerY, radius, gemGrupp):
    paletHSV = cv2.cvtColor(palette, cv2.COLOR_BGR2HSV)
    alpha = int(360 / numberParts)
    startAngle = alpha/2
    endAngle = alpha + startAngle


    r = int(radius / 2)
    vector = []

    vector.append('{0}'.format(gemGrupp))

    for i in range(numberParts):
        x = int(centerX + r * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
        y = int(centerY + r * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))
        h, s, v = np.uint8(paletHSV[x, y])
        colorCode = 1000000 * h + 1000 * s + v
        vector.append('{0}'.format(colorCode))
        startAngle = startAngle + alpha
        endAngle = endAngle + alpha

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

    row = 0
    columns = []

    # подготовка таблицы
    for j in range(0, numberParts+1):
        columns.append('{0}'.format(j))

    array = pd.DataFrame(columns=columns)

    for i in range(numberGem):
        gem = cv2.imread(nameImage[i])
        vector = maskCreation(gem, 4, 1)

        maxInd = numberParts + 1
        array.loc[row] = vector
        row = row + 1

    print(array)




# main

nameOne = "./image/"
nameTwo = ".jpg"

name = []
for i in range(1,3):
    temp = nameOne + str(i) + nameTwo
    name.append(temp)

gruppCreation(name, 4, 1)


"""
A = [14, 15, 16, 17, 18]
b = [19, 20, 21, 22, 23]

columns = []

for i in range(0, 5):
    columns.append('{0}'.format(i))
    df = pd.DataFrame(columns=columns)
row = 0
a = []
for i in range(0, 5):
    a.append('{0}'.format(A[i]))

df.loc[row] = a


print("test")
print(df)


df.to_csv("./test.csv", index=None, header=True)
"""

"""
gem = cv2.imread("./image/1.jpg")
palette = maskCreation(gem, 4, 1)
print("palette")
print(palette)


"""