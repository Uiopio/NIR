import cv2
import numpy as np
import math
import time
import pandas as pd
from lib.gem import Gem






if __name__ == "__main__":


    startTime = time.time()

    # Группа 1
    nameOne = "./inputImage/"
    nameTwo = ".jpg"
    name = []
    for groupp in range(1, 4):
        for i in range(1, 10):
            temp = nameOne + str(groupp) + "_" + str(i) + nameTwo
            name.append(temp)

    print(name)

    columns = []
    for i in range(0, 9):
        columns.append('{0}'.format(i))

    array = pd.DataFrame(columns=columns)

    groupGem1_list = []
    for i in range(0, 9):
        groupGem1_list.append(Gem(gemId=1, numParts=4, inputImageGem=cv2.imread(name[i])))
        groupGem1_list[i].returnVector()
        array.loc[i] = groupGem1_list[i].gemColorVector

    groupGem2_list = []
    for i in range(0, 9):
        groupGem1_list.append(Gem(gemId=2, numParts=4, inputImageGem=cv2.imread(name[i])))
        groupGem1_list[i + 9].returnVector()
        array.loc[i + 9] = groupGem1_list[i+9].gemColorVector

    groupGem3_list = []
    for i in range(0, 9):
        groupGem1_list.append(Gem(gemId=3, numParts=4, inputImageGem=cv2.imread(name[i])))
        groupGem1_list[i + 18].returnVector()
        array.loc[i + 18] = groupGem1_list[i +18].gemColorVector


    array.to_csv("./result.csv", index=None, header=True)
    print(time.time()- startTime)






