import os

import numpy
import pandas as pd
from sklearn import tree

from common import confusion_matrix
from common import fileutils
from common import metrics

def predict_base_dt(input_data, validation_data, test_data, filename):
    inputX = input_data.iloc[:,0:1024] # Binary features (first 1024 columns)
    inputY = input_data.iloc[:,1024] # Index representing the class (last column)

    # Create a decision tree fit using the training data
    basedt = tree.DecisionTreeClassifier()
    basedt.fit(inputX, inputY)

    # Validate decision tree. But we aren't calibrating any parameters so this is optional.
    validX = validation_data.iloc[:, 0:1024]
    validY = validation_data.iloc[:,1024]
    score = basedt.score(validX, validY)
    print(f'Score on validation set: {score}')

    # Predict using the test data
    testX = test_data.iloc[:,0:1024]
    testY = test_data.iloc[:,1024]
    predictions = basedt.predict(testX)
    score2 = basedt.score(testX, testY)
    
    # Print the predictions to CSV
    output_df = pd.DataFrame(predictions)
    output_df.transpose()
    fileutils.write_output(filename, output_df)

    return predictions, testY


print("===Base-DT Data Set 1===")
input1 = fileutils.load_csv("train_1")
valid1 = fileutils.load_csv("val_1")
test1 = fileutils.load_csv("test_with_label_1")

test_predictions1, test_correct1 = predict_base_dt(input1, valid1, test1, "Base-DT-DS1")

confusion_matrix.create_alphabet(test_predictions1, test_correct1)
metrics.compute(test_predictions1, test_correct1, 26)

print("===Base-DT Data Set 2===")
input2 = fileutils.load_csv("train_2")
valid2 = fileutils.load_csv("val_2")
test2 = fileutils.load_csv("test_with_label_2")

test_predictions2, test_correct2 = predict_base_dt(input2, valid2, test2, "Base-DT-DS2")

confusion_matrix.create_greek(test_predictions2, test_correct2)
metrics.compute(test_predictions2, test_correct2, 10)