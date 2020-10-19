import os

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from common import confusion_matrix
from common import fileutils
from common import metrics
from common import alphabet

def predict_best_dt(input_data, validation_data, test_data, filename):
    inputX = input_data.iloc[:, 0:1024]  # Binary features (first 1024 columns)
    inputY = input_data.iloc[:, 1024]  # Index representing the class (last column)

    grid_parameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10],
        'min_samples_split': [0.01, 0.1, 0.5, 1.0],
        'min_impurity_decrease': [0.0, 0.1, 0.2, 0.5, 1.0],
        'class_weight': [None, 'balanced']
    }

    bestdt = GridSearchCV(DecisionTreeClassifier(), grid_parameters, verbose=3)
    bestdt.fit(inputX, inputY)

    # Validate decision tree
    validX = validation_data.iloc[:, 0:1024]
    validY = validation_data.iloc[:,1024]
    score = bestdt.score(validX, validY)
    print(f'Score on validation set: {score}')

    # Predict using the test data
    testX = test_data.iloc[:,0:1024]
    testY = test_data.iloc[:,1024]
    predictions = bestdt.predict(testX)

    # Print the predictions to CSV
    output_df = pd.DataFrame(predictions)
    output_df.transpose()
    fileutils.write_output(filename, output_df)

    return predictions, testY

print("===Best-DT Data Set 1===")
input1 = fileutils.load_csv("train_1")
valid1 = fileutils.load_csv("val_1")
test1 = fileutils.load_csv("test_with_label_1")

test_predictions1, test_correct1 = predict_best_dt(input1, valid1, test1, "Best-DT-DS1")

confusion_matrix.create(test_predictions1, test_correct1, alphabet.latin(), "Best-DT-DS1")
metrics.compute(test_predictions1, test_correct1, alphabet.latin(), "Best-DT-DS1")

print("===Best-DT Data Set 2===")
input2 = fileutils.load_csv("train_2")
valid2 = fileutils.load_csv("val_2")
test2 = fileutils.load_csv("test_with_label_2")

test_predictions2, test_correct2 = predict_best_dt(input2, valid2, test2, "Best-DT-DS2")

confusion_matrix.create(test_predictions2, test_correct2, alphabet.greek(), "Best-DT-DS2")
metrics.compute(test_predictions2, test_correct2, alphabet.greek_no_unicode(), "Best-DT-DS2")
