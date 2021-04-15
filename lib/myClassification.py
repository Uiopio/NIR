import pandas as pd


tab = pd.read_csv("result.csv")


print(tab.loc[0:2])
list = tab.loc[0]
print(list)