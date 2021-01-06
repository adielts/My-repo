import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def execut_algorithm(algorithm, X_train_validation, X_test, Y_train_validation, Y_test, efficient_c, two_attributes):
    lr = LogisticRegression(C=efficient_c, penalty='l1', solver='saga', multi_class='multinomial', tol=0.001,
                            max_iter=len(X_train_validation))
    lr.fit(X_train_validation[:, two_attributes], Y_train_validation)
    two_attrb_acc = lr.score(X_test[:, two_attributes], Y_test)
    print('Accuracy for Test part is with 2 best attributes:', two_attrb_acc)

    # Y_predict = lr.predict(X_test[:, two_attributes])
    # confusion_matrix_general = np.zeros((10, 10))
    # for i in range(len(Y_test)):
    #     confusion_matrix_general[Y_predict[i]][Y_test[i]] += 1
    # sn.heatmap(confusion_matrix_general, annot=True, xticklabels=np.arange(10), yticklabels=np.arange(10))
    # plt.xlabel('Predict')
    # plt.ylabel('Actual')
    # plt.title(algorithm + '\'s confusion matrix')
    # plt.show()
    return two_attrb_acc


def lasso_find_2_efficient_attributs(X_train_validation, X_test, Y_train_validation, Y_test, efficient_c):
    lasso_lr = LogisticRegression(C=efficient_c, penalty='l1', solver='saga', multi_class='multinomial', tol=0.001,
                                  max_iter=len(X_train_validation))
    lasso_lr.fit(X_train_validation, Y_train_validation)
    accuracy = lasso_lr.score(X_test, Y_test)
    print('Accuracy for Test part is with selected C:', accuracy)
    sums = np.sum(np.absolute(lasso_lr.coef_), axis=0)
    best_two = np.argsort(-sums)[:2]
    print('\n****** Lasso ******')
    print('Efficient Attributes:', best_two[0], best_two[1])
    # Y_predict = lasso_lr.predict(X_test)
    # confusion_matrix_general = np.zeros((10, 10))
    # for i in range(len(Y_test)):
    #     confusion_matrix_general[Y_predict[i]][Y_test[i]] += 1
    # sn.heatmap(confusion_matrix_general, annot=True, xticklabels=np.arange(10), yticklabels=np.arange(10))
    # plt.xlabel('Predict')
    # plt.ylabel('Actual')
    # plt.title('Logistic Regression\'s confusion matrix')
    # plt.show()
    return [best_two, accuracy]


def greedy_find_2_efficient_attributs(X_train, X_validation, Y_train, Y_validation, efficient_c):
    greedy_lr = LogisticRegression(C=efficient_c, penalty='l1', solver='saga', multi_class='multinomial', tol=0.001,
                                   max_iter=len(X_train))
    attributes = np.zeros(2).astype(int)
    efficient_attribute = 0
    accuracy = 0
    best_accuracy = 0
    for attribute in range(64):
        greedy_lr.fit(X_train[:, [attribute]], Y_train)
        accuracy = greedy_lr.score(X_validation[:, [attribute]], Y_validation)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            efficient_attribute = attribute
    attributes[0] = efficient_attribute

    for attribute in range(64):
        if attribute != attributes[0]:
            greedy_lr.fit(X_train[:, [attributes[0], attribute]], Y_train)
            accuracy = greedy_lr.score(X_validation[:, [attributes[0], attribute]], Y_validation)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                efficient_attribute = attribute
    attributes[1] = efficient_attribute
    print('\n****** Greedy ******')
    print('Efficient Attributes:', attributes[0], attributes[1])

    return attributes


def find_best_c_logistic_regression(X_train, X_validation, Y_train, Y_validation):
    c = 0.0001
    efficient_c = c
    accuracy = 0
    best_accuracy = 0
    Cs = np.zeros(10)
    accuracies = np.zeros(10)
    for i in range(10):
        lr = LogisticRegression(C=c, penalty='l1', solver='saga', multi_class='multinomial', tol=0.001,
                                max_iter=len(X_train))
        lr.fit(X_train, Y_train)
        accuracy = lr.score(X_validation, Y_validation)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            efficient_c = c
        print('C =', c)
        accuracies[i] = accuracy
        Cs[i] = c
        print('Accuracy', i, accuracy, '\n')
        c = c * 10


    print('Efficient C:', efficient_c)
    print('Accuracy for Validation part is with selected C:', best_accuracy)
    # clrs = ['grey' if (x < max(accuracies)) else 'green' for x in accuracies]
    # plt.bar(np.arange(10), 100 * accuracies, color=clrs)
    # plt.xticks(np.arange(10), Cs)
    # plt.ylabel('Accuracies (%)')
    # plt.xlabel("C's")
    # plt.title('Accuracy depends C')
    # plt.show()
    return efficient_c


def split(X_digits, Y_digits):
    X_train_validation, X_test, Y_train_validation, Y_test = train_test_split(X_digits, Y_digits, test_size=0.1,
                                                                              random_state=42)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_validation, Y_train_validation,
                                                                    test_size=0.33, random_state=42)
    return [X_train, X_validation, X_test, Y_train, Y_validation, Y_test]


if __name__ == '__main__':
    X_digits, Y_digits = load_digits(return_X_y=True)
    X_digits = preprocessing.scale(X_digits)
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = split(X_digits, Y_digits)
    efficient_c = find_best_c_logistic_regression(X_train, X_validation, Y_train, Y_validation)
    best_two_attributes, general_accuracy = lasso_find_2_efficient_attributs(np.concatenate((X_train, X_validation)),
                                                                             X_test,
                                                                             np.concatenate((Y_train, Y_validation)),
                                                                             Y_test, efficient_c)
    lasso_2_att_acc = execut_algorithm('Lasso', np.concatenate((X_train, X_validation)), X_test,
                                       np.concatenate((Y_train, Y_validation)),
                                       Y_test, efficient_c, best_two_attributes)
    best_two_attributes = greedy_find_2_efficient_attributs(X_train, X_validation, Y_train, Y_validation, efficient_c)
    greedy_2_att_acc = execut_algorithm('Greedy', np.concatenate((X_train, X_validation)), X_test,
                                        np.concatenate((Y_train, Y_validation)),
                                        Y_test, efficient_c, best_two_attributes.astype(int))
    # plt.bar(np.arange(3), [100 * lasso_2_att_acc, 100 * greedy_2_att_acc, 100 * general_accuracy])
    # plt.xticks(np.arange(3), ['Lasso', 'Greedy', 'Logistic Regression'])
    # plt.ylabel('Accuracies (%)')
    # plt.xlabel("Algorithms")
    # plt.title('Algorithms\' accuracy compare')
    # plt.show()
