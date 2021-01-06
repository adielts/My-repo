from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#   Build the One vs All matrix using the infrastructures that created.
def build_oneVSall_matrix(LR_Models, x_test, y_test):
    probability_of_Xi = np.zeros(len(LR_Models))
    confusion_matrix = np.zeros((len(LR_Models), len(LR_Models)))
    for j in range(0, x_test.shape[0]):
        for i in range(0, len(LR_Models)):
            probability_of_Xi[i] = (LR_Models[i].predict_proba([x_test[j]]))[0][1]
        best_prob_index = np.argmax(probability_of_Xi)
        confusion_matrix[y_test[j] - 1][best_prob_index] += 1
    sn.heatmap(confusion_matrix, annot=True, xticklabels=('car', 'fad', 'mas', 'gla', 'con', 'adi'),
               yticklabels=('car', 'fad', 'mas', 'gla', 'con', 'adi'))
    accuracy = 0
    for i in range(6):
        accuracy += confusion_matrix[i][i]
    accuracy = accuracy / len(x_test)
    print('One vs All Accuracy: ', accuracy)
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    plt.title('One vs All')
    plt.show()


# Setup the infrastructures for the One vs All LR method
def setupLRsOneVsAll(x_train, y_train):
    logistic_regression_for_car = LogisticRegression(max_iter=1000, class_weight='balanced')
    logistic_regression_for_fad = LogisticRegression(max_iter=1000, class_weight='balanced')
    logistic_regression_for_mas = LogisticRegression(max_iter=1000, class_weight='balanced')
    logistic_regression_for_gla = LogisticRegression(max_iter=1000, class_weight='balanced')
    logistic_regression_for_con = LogisticRegression(max_iter=1000, class_weight='balanced')
    logistic_regression_for_adi = LogisticRegression(max_iter=1000, class_weight='balanced')

    car_y_train = replaceClassName(1, y_train).astype('int')
    fad_y_train = replaceClassName(2, y_train).astype('int')
    mas_y_train = replaceClassName(3, y_train).astype('int')
    gla_y_train = replaceClassName(4, y_train).astype('int')
    con_y_train = replaceClassName(5, y_train).astype('int')
    adi_y_train = replaceClassName(6, y_train).astype('int')

    logistic_regression_for_car.fit(x_train, car_y_train)
    logistic_regression_for_fad.fit(x_train, fad_y_train)
    logistic_regression_for_mas.fit(x_train, mas_y_train)
    logistic_regression_for_gla.fit(x_train, gla_y_train)
    logistic_regression_for_con.fit(x_train, con_y_train)
    logistic_regression_for_adi.fit(x_train, adi_y_train)

    return [logistic_regression_for_car, logistic_regression_for_fad, logistic_regression_for_mas,
            logistic_regression_for_gla, logistic_regression_for_con, logistic_regression_for_adi]


#   Isolate specific class by define it as '1' and others to '0'
def replaceClassName(oneOption, Ys_train):
    copy = Ys_train.copy()
    copy[copy != oneOption] = 0
    copy[copy == oneOption] = 1
    return copy


# Convert all Y vector's values to numbers
def setupY(y):
    y[y == 'car'] = 1
    y[y == 'fad'] = 2
    y[y == 'mas'] = 3
    y[y == 'gla'] = 4
    y[y == 'con'] = 5
    y[y == 'adi'] = 6
    return y


#   Filter 2 classes from array of classes
def filterClasses(x_train, y_train, class1, class2):
    class1Indexes = np.asarray(np.where(y_train == class1))
    class2Indexes = np.asarray(np.where(y_train == class2))
    mergeIndexes = np.sort(np.concatenate((class1Indexes[0], class2Indexes[0])))
    return [x_train[mergeIndexes, :], replaceClassName(class1, y_train[mergeIndexes])]


# Setup the infrastructures for the One vs One LR method
def setupLRsOneVsOne(x_train, y_train, numOfClasses):
    oneVSoneLRsArray = np.zeros((numOfClasses, numOfClasses), dtype=object)
    for j in range(numOfClasses):
        for i in range(j + 1, numOfClasses):
            x_train_ij, y_train_ij = filterClasses(deepcopy(x_train), y_train.copy(), j + 1, i + 1)
            oneVSoneLRsArray[j][i] = LogisticRegression(max_iter=1000, class_weight='balanced').fit(x_train_ij,
                                                                                                    list(y_train_ij))
    return oneVSoneLRsArray


# Execute the One vs One LRs algorithm
def calculate_oneVSoneLR(oneVSoneLRsArray, numOfClasses, x_test, y_test):
    oneVSone_confusion_matrix = np.zeros((numOfClasses, numOfClasses))
    LR_results = np.zeros(numOfClasses)
    for k in range(len(y_test)):
        for j in range(numOfClasses):
            for i in range(numOfClasses):
                if i > j:
                    LR_results[j] += (oneVSoneLRsArray[j][i].predict_proba([x_test[k]]))[0][1]
                elif i < j:
                    LR_results[j] += (oneVSoneLRsArray[i][j].predict_proba([x_test[k]]))[0][0]
        oneVSone_confusion_matrix[y_test[k] - 1][np.argmax(LR_results)] += 1
        LR_results = np.zeros(numOfClasses)

    sn.heatmap(oneVSone_confusion_matrix, annot=True, xticklabels=('car', 'fad', 'mas', 'gla', 'con', 'adi'),
               yticklabels=('car', 'fad', 'mas', 'gla', 'con', 'adi'))
    accuracy = 0
    for i in range(6):
        accuracy += oneVSone_confusion_matrix[i][i]
    accuracy = accuracy / len(x_test)
    print('One vs One Accuracy: ', accuracy)
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    plt.title('One vs One')
    plt.show()


if __name__ == '__main__':
    excelData = pd.read_excel('breastTissue for one and all.xlsx', sheet_name='Data')
    df = pd.DataFrame(excelData,
                      columns=['Case #', 'Class', 'I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP', 'DR', 'P'])
    x = df[['I0', 'PA500', 'HFS', 'DA', 'Area', 'A/DA', 'Max IP', 'DR', 'P']]
    y = df['Class']
    y = y.to_numpy()
    numericY = setupY(y.copy())
    x = preprocessing.scale(x)
    x_train, x_test, y_train, y_test = train_test_split(x, numericY, test_size=0.33, random_state=42)
    LR_Models = setupLRsOneVsAll(x_train, y_train)
    build_oneVSall_matrix(LR_Models, x_test, y_test)

    oneVSoneLRsArray = setupLRsOneVsOne(x_train, y_train, 6)
    calculate_oneVSoneLR(oneVSoneLRsArray, 6, x_test, y_test)
