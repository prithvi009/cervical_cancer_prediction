import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load data
print("Dataset Manipulation")
print("Reading data from text file")

data = []
with open('dataset.txt', 'r') as input_file:
    next(input_file)
    for line in input_file:
        row = line.strip().split(',')
        data.append(row)

print("Data loaded")
print("Number of instances:", len(data))
print("Number of features:", len(data[0]))

# Clean data
for i in range(len(data)):
    data[i][-1] = data[i][-1].strip()

print("Replace missing values '?' with nan")
data_numpy = np.asarray(data)
data_numpy[data_numpy == '?'] = np.nan
data_numpy = data_numpy.astype(float)

print("Replace nan with mean of column")
nan_indices = np.argwhere(np.isnan(data_numpy))
col_mean = np.nanmean(data_numpy, axis=0)
data_numpy[nan_indices[:, 0], nan_indices[:, 1]] = col_mean[nan_indices[:, 1]]

print("Extracting target from dataset")
y = data_numpy[:, -1]

# Split data into 80% train, 20% test
print("Splitting dataset")
x_train, x_test, y_train, y_test = train_test_split(data_numpy[:, :-1], y, test_size=0.2, random_state=0)

# Data Preprocessing
print("Data Preprocessing")
zscore_train_numpy = stats.zscore(x_train, axis=1)
zscore_test_numpy = stats.zscore(x_test, axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_numpy = scaler.fit_transform(x_train)
scaled_test_numpy = scaler.fit_transform(x_test)

scaler = StandardScaler()
standardized_train_numpy = scaler.fit_transform(x_train)
standardized_test_numpy = scaler.fit_transform(x_test)

scaler = Normalizer()
normalized_train_numpy = scaler.fit_transform(x_train)
normalized_test_numpy = scaler.fit_transform(x_test)

scaler = Binarizer(threshold=0.0)
binarized_train_numpy = scaler.fit_transform(x_train)
binarized_test_numpy = scaler.fit_transform(x_test)

# Feature Selection
print("Feature Selection")
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
variance_train_numpy = sel.fit_transform(x_train)
variance_test_numpy = sel.fit_transform(x_test)

kbest_train_numpy = SelectKBest(chi2, k=5).fit_transform(x_train, y_train)
kbest_test_numpy = SelectKBest(chi2, k=5).fit_transform(x_test, y_test)

print("Linear Discriminant Analysis")
lda = LinearDiscriminantAnalysis()
lda.fit(kbest_train_numpy, y_train)
predict_1 = lda.predict(kbest_test_numpy)
score = accuracy_score(y_test, predict_1)
print("Linear Discriminant Analysis Accuracy:", score * 100, "%")

print("Quadratic Discriminant Analysis")
qda = QuadraticDiscriminantAnalysis()
qda.fit(kbest_train_numpy, y_train)
predict_2 = qda.predict(kbest_test_numpy)
score = accuracy_score(y_test, predict_2)
print("Quadratic Discriminant Analysis Accuracy:", score * 100, "%")

print("Logistic Regression")
logisticRegr = LogisticRegression()
logisticRegr.fit(kbest_train_numpy, y_train)
predict_3 = logisticRegr.predict(kbest_test_numpy)
score = accuracy_score(y_test, predict_3)
print("Logistic Regression Accuracy:", score * 100, "%")

print("Gaussian Naive Bayes")
GaussNB = GaussianNB()
GaussNB.fit(kbest_train_numpy, y_train)
predict_4 = GaussNB.predict(kbest_test_numpy)
score = accuracy_score(y_test, predict_4)
print("Gaussian Naive Bayes Accuracy:", score * 100, "%")
