from builtins import len
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def plotJTetas(jTetas, labels, ks, title):
    for i in range(0, len(jTetas)):
        plt.plot(np.arange(ks), jTetas[i], label=labels[i])

    plt.ylabel('J(teta)')
    plt.xlabel("K's")
    plt.title(title)
    plt.legend()
    plt.show()


def adagrad(vectorTeta, matrix, vectorY, alfa, epsilon, maxK):
    k = 0
    stopCondition = 10
    jTetasArryForPlot = np.zeros(maxK)
    gArray = np.zeros(vectorTeta.shape)
    newGArray = np.zeros(vectorTeta.shape)
    vectorAlfa = np.zeros(vectorTeta.shape)

    while stopCondition > 0.000000001 and k < maxK:
        gradient = gradientJCalculation(vectorTeta, matrix, vectorY)
        for j in range(0, len(vectorTeta)):
            newGArray[j] = gArray[j] + np.square(gradient[j])
            vectorAlfa[j] = alfa / (np.sqrt(newGArray[j] + epsilon))

        jTetasArryForPlot[k] = jTetaCalculation(vectorTeta, matrix, vectorY)
        vectorTetaKPlus1 = vectorTeta - (vectorAlfa * gradientJCalculation(vectorTeta, matrix, vectorY))
        stopCondition = np.linalg.norm(vectorTetaKPlus1 - vectorTeta)
        vectorTeta = vectorTetaKPlus1
        gArray = newGArray
        k += 1

    return [vectorTeta, jTetasArryForPlot]


def miniBatch(vectorTeta, matrix, vectorY, alfa, t, maxK):
    k = 0
    sum = 0
    stopcondition = 10
    jTetasArryForPlot = np.zeros(maxK)
    m = matrix.shape[0]
    n = m / t
    vectorTetaKPlus1 = np.zeros((len(vectorTeta), 1))

    while stopcondition > 0.0001 and k < maxK:
        jTetasArryForPlot[k] = jTetaCalculation(vectorTeta, matrix, vectorY)
        for j in range(0, len(vectorTeta)):
            for i in range(int((k * n) % m), int((((k + 1) * n) - 1) % m)):
                sum += alfa * (matrix[i][j] * (hTetaCalculation(vectorTeta, [matrix[i]]) - vectorY[i]))
            sum = sum / (int((((k + 1) * n) - 1) % m) - int((k * n) % m))
            vectorTetaKPlus1[j] = vectorTeta[j] - sum
        stopcondition = np.absolute(jTetasArryForPlot[k] - jTetasArryForPlot[k - 1])
        vectorTeta = vectorTetaKPlus1
        k += 1

    return [vectorTeta, jTetasArryForPlot]


def gradientDescent(vectorTeta, matrix, vectorY, alfa, maxK):
    k = 0
    stopCondition = 10
    jTetasArryForPlot = np.zeros(maxK)

    while stopCondition > 0.0001 and k < maxK:
        jTetasArryForPlot[k] = jTetaCalculation(vectorTeta, matrix, vectorY)
        vectorTetaKPlus1 = vectorTeta - (alfa * gradientJCalculation(vectorTeta, matrix, vectorY))
        stopCondition = np.linalg.norm(vectorTetaKPlus1 - vectorTeta)
        vectorTeta = vectorTetaKPlus1
        k += 1

    return [vectorTeta, jTetasArryForPlot]


def gradientJCalculation(vectorTeta, matrix, vectorY):
    gradientJ = (np.transpose(matrix) @ ((matrix @ vectorTeta) - vectorY)) / len(vectorY)
    return gradientJ


def jTetaCalculation(vectorTeta, matrix, vectorY):
    m = len(vectorY)
    sum = 0
    for i in range(0, m):
        currentHTeta = hTetaCalculation(vectorTeta, [matrix[i]])
        sum += np.square(currentHTeta - vectorY[i])

    return (1 / (2 * m)) * sum


def hTetaCalculation(vectorTeta, vectorX):
    hTeta = vectorX @ vectorTeta
    return hTeta[0][0]


def createNewStandardizedMatrix(matrix, xsAvareges, sigma):
    matrixCopy = deepcopy(matrix)
    for column in range(0, matrix.shape[1]):
        for row in range(0, matrix.shape[0]):
            matrixCopy[row][column] = (matrix[row][column] - xsAvareges[column]) / sigma[column]
    eOfX, sigma = findAvarege(matrixCopy)
    print('Validation check:')
    print('E of X:', eOfX)
    print('Sigma (Stiyat Teken):', sigma)

    onesVector = np.ones((matrix.shape[0], 1))
    matrixCopy = np.hstack((onesVector, matrixCopy))
    print('normalized matrix: \n', matrixCopy)
    return matrixCopy


def normalizedYVector(vectorY):
    yAvarege, sigma = findAvarege(vectorY)
    for row in range(0, vectorY.shape[0]):
        vectorY[row][0] = (vectorY[row][0] - yAvarege) / sigma
    return vectorY


def findAvarege(matrix):
    xsAvareges = np.mean(matrix, axis=0)  # E(x)
    sigma = np.std(matrix, axis=0)  # stiyat teken
    return [xsAvareges, sigma]


def importDataFromExcel():
    df = np.genfromtxt('cancer_data.csv', delimiter=',')

    # get the Xs matrix
    matrix = df[:, :len(df[0]) - 1]

    # get the Y vector
    vectorY = df[:, len(df[0]) - 1:len(df[0])]
    return [matrix, vectorY]


if __name__ == '__main__':
    matrix, vectorY = importDataFromExcel()
    xsAvareges, sigma = findAvarege(matrix)
    normalizedMatrix = createNewStandardizedMatrix(matrix, xsAvareges, sigma)
    vectorY = normalizedYVector(vectorY)
    # hTetaCalculation()
    # jTeta = jTetaCalculation(np.zeros((normalizedMatrix.shape[1], 1)), normalizedMatrix, vectorY)
    # gradientJCalculation(np.zeros((normalizedMatrix.shape[1], 1)), normalizedMatrix, vectorY)
    maxK = 30

    # --------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------- implement G.D / Vanilla ------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------

    print('Run Gradiend Descent - Vanilla algorithm')
    finalGdTetas1, gdTetasPlot1 = gradientDescent(np.zeros((normalizedMatrix.shape[1], 1)), normalizedMatrix, vectorY,
                                                  0.1, maxK)
    finalGdTetas2, gdTetasPlot2 = gradientDescent(np.zeros((normalizedMatrix.shape[1], 1)), normalizedMatrix, vectorY,
                                                  0.01, maxK)
    finalGdTetas3, gdTetasPlot3 = gradientDescent(np.zeros((normalizedMatrix.shape[1], 1)), normalizedMatrix, vectorY,
                                                  0.001, maxK)
    plotJTetas(np.array([gdTetasPlot1, gdTetasPlot2, gdTetasPlot3]),
               ["Vanilla's alfa = 0.1", "Vanilla's alfa = 0.01", "Vanilla's alfa = 0.001"], maxK,
               'Error price function Gradiend Descent - Vanilla')

    # --------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------- implement Mini Batch ---------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    print('Run Mini Batch algorithm')
    finalMinibtchTetas, minibtchTetasPlot = miniBatch(np.zeros((normalizedMatrix.shape[1], 1)), normalizedMatrix,
                                                      vectorY,
                                                      0.1, 100, maxK)
    plotJTetas(np.array([gdTetasPlot1, minibtchTetasPlot]),
               ["Vanilla's alfa = 0.1", "Mini-Batch's alfa = 0.1"], maxK,
               'Error price function Vanilla / Mini-Batch')

    # --------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------- implement AdaGrad --------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    print('Run AdaGrad algorithm')
    finalAdaGradTetas, adaGradTetasPlot = adagrad(np.zeros((normalizedMatrix.shape[1], 1)), normalizedMatrix,
                                                  vectorY,
                                                  0.1, 0.0000001, maxK)
    plotJTetas(np.array([gdTetasPlot1, minibtchTetasPlot, adaGradTetasPlot]),
               ["Vanilla's alfa = 0.1", "Mini-Batch's alfa = 0.1", "AdaGrad's alfa = 0.1"], maxK,
               'Error price function Vanilla / Mini-Batch / AdaGrad')
