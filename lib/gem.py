import cv2
import numpy as np
import math
import pandas as pd




class Gem:
    # Класс создающий вектора и заполняющий поля gem

    def __init__(self, gemId, numParts, inputImageGem):
        self.gemId = gemId  # Номер группы камня. None если нужно определеит группу камня
        self.gemImage = inputImageGem  # Изображение камня
        self.numberParts = numParts  # Количество ячеек для создания вектора
        self.gemSize = None  # Размер камня
        self.gemPosition = None
        self.gemColorVector = None  # вектор цветов

    # Возвращает вектор цветов камня в HSV 3 канала
    def returnVectorHSV3(self):
        mask = self.__maskCreation()
        palette = self.__paletteCreation(mask)
        self.gemColorVector = self.__vectorCreationHSV3(palette)

    # Возвращает вектор цветов камня в HSV 2 канала
    def returnVectorHSV2(self):
        mask = self.__maskCreation()
        palette = self.__paletteCreation(mask)
        self.gemColorVector = self.__vectorCreationHSV2(palette)

    # Возвращает вектор цветов камня в RGB 3 канала
    def returnVectorRGB3(self):
        # Возвращает вектор цветов камня
        mask = self.__maskCreation()
        palette = self.__paletteCreation(mask)
        self.gemColorVector = self.__vectorCreationRGB3(palette)


    def __vectorCreationRGB3(self, palette):
        # Созздает вектор цветов
        alpha = int(360 / self.numberParts)
        startAngle = alpha / 2 + 2
        endAngle = alpha + startAngle

        centerY = int(1280 / 2)
        centerX = int(720 / 2)

        r1 = 50
        r2 = 200
        colorVector = []

        colorVector.append('{0}'.format(self.gemId))

        for i in range(self.numberParts):
            x = int(centerX + r1 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
            y = int(centerY + r1 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))

            b, g, r = np.uint8(palette[x, y])
            colorCode = 1000000 * b + 1000 * g + r
            colorVector.append('{0}'.format(colorCode))
            startAngle = startAngle + alpha
            endAngle = endAngle + alpha

        for i in range(self.numberParts):
            x = int(centerX + r2 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
            y = int(centerY + r2 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))

            b, g, r = np.uint8(palette[x, y])
            colorCode = 1000000 * b + 1000 * g + r
            colorVector.append('{0}'.format(colorCode))
            startAngle = startAngle + alpha
            endAngle = endAngle + alpha

        return colorVector


    def __vectorCreationHSV2(self, palette):
        # Созздает вектор цветов
        paletHSV = cv2.cvtColor(palette, cv2.COLOR_BGR2HSV)
        alpha = int(360 / self.numberParts)
        startAngle = alpha / 2 + 2
        endAngle = alpha + startAngle

        centerY = int(1280 / 2)
        centerX = int(720 / 2)

        r1 = 50
        r2 = 200
        colorVector = []

        colorVector.append('{0}'.format(self.gemId))

        for i in range(self.numberParts):
            x = int(centerX + r1 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
            y = int(centerY + r1 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))
            cv2.circle(palette, (centerY, centerX), 15, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(palette, (y, x), 15, (255, 255, 255), 3, cv2.LINE_AA)
            h, s, v = np.uint8(paletHSV[x, y])

            colorCode = 1000 * h + s
            colorVector.append('{0}'.format(colorCode))
            startAngle = startAngle + alpha
            endAngle = endAngle + alpha

        for i in range(self.numberParts):
            x = int(centerX + r2 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
            y = int(centerY + r2 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))
            cv2.circle(palette, (centerY, centerX), 15, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(palette, (y, x), 15, (255, 255, 255), 3, cv2.LINE_AA)
            h, s, v = np.uint8(paletHSV[x, y])
            colorCode = 1000 * h + s
            colorVector.append('{0}'.format(colorCode))
            startAngle = startAngle + alpha
            endAngle = endAngle + alpha

        return colorVector



    def __vectorCreationHSV3(self, palette):
        # Созздает вектор цветов
        width = 1280
        height = 720

        paletHSV = cv2.cvtColor(palette, cv2.COLOR_BGR2HSV)
        alpha = int(360 / self.numberParts)
        startAngle = alpha / 2 + 2
        endAngle = alpha + startAngle

        centerY = int(width / 2)
        centerX = int(height / 2)

        r1 = 50
        r2 = 200
        colorVector = []

        colorVector.append('{0}'.format(self.gemId))

        for i in range(self.numberParts):
            x = int(centerX + r1 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
            y = int(centerY + r1 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))
            cv2.circle(palette, (centerY, centerX), 15, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(palette, (y, x), 15, (255, 255, 255), 3, cv2.LINE_AA)
            h, s, v = np.uint8(paletHSV[x, y])

            colorCode = 1000000 * h + 1000 * s + v
            colorVector.append('{0}'.format(colorCode))
            startAngle = startAngle + alpha
            endAngle = endAngle + alpha

        for i in range(self.numberParts):
            x = int(centerX + r2 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
            y = int(centerY + r2 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))
            cv2.circle(palette, (centerY, centerX), 15, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(palette, (y, x), 15, (255, 255, 255), 3, cv2.LINE_AA)
            h, s, v = np.uint8(paletHSV[x, y])
            colorCode = 1000000 * h + 1000 * s + v
            colorVector.append('{0}'.format(colorCode))
            startAngle = startAngle + alpha
            endAngle = endAngle + alpha

        return colorVector



    def __paletteCreation(self, gemAndMask):
        # Создает палитру основаных цветов камня
        width = 1280
        height = 720
        palette = np.zeros((720, 1280, 3), np.uint8)

        alpha = int(360 / self.numberParts)
        startAngle = 0
        endAngle = alpha

        center = [int(1280 / 2), int(720 / 2)]

        radius = self.gemSize
        centerX = self.gemPosition[1]
        centerY = self.gemPosition[0]

        r1 = int(radius / 4)
        r2 = int(radius - radius / 6)
        newR1 = r1
        newR2 = r2
        axes1 = (250, 250)
        axes2 = (100, 100)

        for i in range(self.numberParts):
            x = int(centerX + newR1 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
            y = int(centerY + newR1 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))

            b, g, r = np.uint8(gemAndMask[x, y])

            cv2.ellipse(palette, (center[0], center[1]), axes1, 0, int(startAngle), int(endAngle),
                        (int(b), int(g), int(r)), -1)
            startAngle = startAngle + alpha
            endAngle = endAngle + alpha


        for i in range(self.numberParts):
            x = int(centerX + newR2 * math.sin(((startAngle + endAngle) / 2) * math.pi / 180))
            y = int(centerY + newR2 * math.cos(((startAngle + endAngle) / 2) * math.pi / 180))

            b, g, r = np.uint8(gemAndMask[x, y])
            cv2.ellipse(palette, (center[0], center[1]), axes2, 0, int(startAngle), int(endAngle),
                        (int(b), int(g), int(r)), -1)
            startAngle = startAngle + alpha
            endAngle = endAngle + alpha

        #cv2.imshow("palette", palette)
        #cv2.waitKey()

        return palette



    def __maskCreation(self):
        # Создает изображение с маской
        # разбиение изображения на каналы
        width = 1280
        height = 720
        self.gemImage = cv2.resize(self.gemImage, (width, height))

        (b, g, r) = cv2.split(self.gemImage)

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


        rows = temp.shape[0]
        mask = np.zeros((height, width, 1), np.uint8)

        # поиск кругов (камня)
        circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, rows, param1=120, param2=20, minRadius=50, maxRadius=350)

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(self.gemImage, (i[0], i[1]), int(i[2]), (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(self.gemImage, (i[0], i[1]), 2, (0, 0, 255), 3)

        #cv2.imshow("image", self.gemImage)
        #cv2.waitKey()

        # создание маски по найденному кругу
        circles = np.uint16(np.around(circles))
        i = circles[0, 0]
        center = (i[0], i[1])
        self.gemPosition = center

        radius = i[2]
        self.gemSize = radius

        cv2.circle(mask, center, radius - 30, 255, cv2.FILLED, 8, 0)

        gemAndMask = cv2.bitwise_and(self.gemImage, self.gemImage, mask=mask)
        #cv2.imshow("gemAndMask", gemAndMask)
        #cv2.waitKey()
        return gemAndMask