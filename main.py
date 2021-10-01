import cv2
import numpy as np
import math
import time
import pandas as pd
from lib.gem import Gem
import glob


if __name__ == "__main__":
    startTime = time.time()
    # количество папок\групп
    numGroups = len(glob.glob('./down/*'))

    k = 12

    # Подготовка таблицы
    columns = []
    for i in range(0, (k*2+1)):
        columns.append('{0}'.format(i))

    array = pd.DataFrame(columns=columns)

    stroke = 0

    for i in range(1, numGroups + 1):
        path = './down/' + str(i) + '/*'
        numGems = len(glob.glob(path))

        nameOne = "./down/" + str(i)
        nameTwo = ".jpg"
        #name = []

        for numImage in range(0, numGems):
            stroke = stroke + 1
            temp = nameOne + "/" + str(i) + "_" + str(numImage) + nameTwo
            #name.append(temp)

            gem = Gem(gemId= i, numParts=k, inputImageGem=cv2.imread(temp))
            gem.returnVectorRGB3()
            array.loc[stroke] = gem.gemColorVector

            print("numImage ", numImage)

        #print(i)


    array.to_csv("./RGB3_12.csv", index=None, header=True)
    print(time.time() - startTime)



#########################################################################################################





