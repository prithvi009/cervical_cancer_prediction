import pandas as pd
from data_preprocessing import preprocess_data, split_data
from classification_algorithms import apply_lda, apply_qda, apply_logistic_regression, apply_gaussian_nb, apply_feature_selection, calculate_accuracy


data = pd.read_csv('cervical_cancer_dataset.csv')


preprocessed_data = preprocess_data(data)
x_train, x_test, y_train, y_test = split_data(preprocessed_data)


kbest_train, kbest_test = apply_feature_selection(x_train, x_test, y_train, y_test, k=5)


predict_lda = apply_lda(kbest_train, y_train, kbest_test)
predict_qda = apply_qda(kbest_train, y_train, kbest_test)
predict_logistic = apply_logistic_regression(kbest_train, y_train, kbest_test)
predict_nb = apply_gaussian_nb(kbest_train, y_train, kbest_test)

# Calculate Accuracy
accuracy_lda = calculate_accuracy(y_test, predict_lda)
accuracy_qda = calculate_accuracy(y_test, predict_qda)
accuracy_logistic = calculate_accuracy(y_test, predict_logistic)
accuracy_nb = calculate_accuracy(y_test, predict_nb)

print("Linear Discriminant Analysis Accuracy:", accuracy_lda, "%")
print("Quadratic Discriminant Analysis Accuracy:", accuracy_qda, "%")
print("Logistic Regression Accuracy:", accuracy_logistic, "%")
print("Gaussian Naive Bayes Accuracy:", accuracy_nb, "%")
