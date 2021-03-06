import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import time


"""
#загружаем таблицу
df = pd.read_csv('8_clast.csv')

y = df['0']
df.drop(['0'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.values, y, test_size=0.5) # Разбитие данных на обучающие и тестовые


#мктод ближних соседей
def kNN():
    knn_clf = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
    knn_clf.fit(X_train, y_train)
    predicted = knn_clf.predict(X_test)
    acc = accuracy_score(y_test, predicted)
    print(y_test)
    print(predicted)
    print('acc=', acc)
"""
"""
    # # Поиск лучшего набора параметров и лучшая точность
    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
    knn_params = {'knn__n_neighbors': range(1, 30), 'knn__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
    knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)
    knn_grid.fit(X_train, y_train)
    print('best_params=', knn_grid.best_params_)
    print('best_score_=', knn_grid.best_score_)
"""



def testKNN(name):
    accuracy = []
    average_accuracy = []
    max_acc = [0,0,0,0,0]
    for i in range(5):
        tab = pd.read_csv(name[i])
        y = tab['0']
        tab.drop(['0'], axis=1, inplace=True)


        accuracy.append(i)
        temp_max_acc = 0
        temp = 0
        for j in range(10):
            X_train, X_test, y_train, y_test = train_test_split(tab.values, y, test_size=0.5)
            knn_clf = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
            knn_clf.fit(X_train, y_train)
            predicted = knn_clf.predict(X_test)
            acc = accuracy_score(y_test, predicted)

            if acc > temp_max_acc:
                temp_max_acc = acc
                max_acc[i] = acc

            temp = temp + acc
            accuracy.append(acc)
        temp = temp / 10
        average_accuracy.append(temp)

    print("точность каждой итерации", accuracy)
    print("средняя точность", average_accuracy)
    print("максимальная точноть", max_acc)



"""
name = []
tab1 = "4_clast.csv"
tab2 = "8_clast.csv"
tab3 = "12_clast.csv"
tab4 = "16_clast.csv"
tab5 = "24_clast.csv"

name.append(tab1)
name.append(tab2)
name.append(tab3)
name.append(tab4)
name.append(tab5)
"""

name = []
tab1 = "2_2_clast.csv"
tab2 = "4_2_clast.csv"
tab3 = "8_2_clast.csv"
tab4 = "12_2_clast.csv"
tab5 = "16_2_clast.csv"

name.append(tab1)
name.append(tab2)
name.append(tab3)
name.append(tab4)
name.append(tab5)



testKNN(name)


