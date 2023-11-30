from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score

def apply_lda(x_train, y_train, x_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    predict = lda.predict(x_test)
    return predict

def apply_qda(x_train, y_train, x_test):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x_train, y_train)
    predict = qda.predict(x_test)
    return predict

def apply_logistic_regression(x_train, y_train, x_test):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    predict = logisticRegr.predict(x_test)
    return predict

def apply_gaussian_nb(x_train, y_train, x_test):
    GaussNB = GaussianNB()
    GaussNB.fit(x_train, y_train)
    predict = GaussNB.predict(x_test)
    return predict

def apply_feature_selection(x_train, x_test, y_train, y_test, k=5):
    kbest_train = SelectKBest(chi2, k=k).fit_transform(x_train, y_train)
    kbest_test = SelectKBest(chi2, k=k).fit_transform(x_test, y_test)
    return kbest_train, kbest_test

def calculate_accuracy(y_test, predict):
    return accuracy_score(y_test, predict) * 100
