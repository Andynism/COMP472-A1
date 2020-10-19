import os

import pandas as pd
from sklearn.neural_network import MLPClassifier

from common import confusion_matrix
from common import fileutils
from common import metrics
from common import alphabet

def predict_base_mlp(input_data, validation_data, test_data, filename):
    inputX = input_data.iloc[:,0:1024] # Binary features (first 1024 columns)
    inputY = input_data.iloc[:,1024] # Index representing the class (last column)

    # Create a decision tree fit using the training data
    basemlp = MLPClassifier(100,activation='logistic', solver='sgd', max_iter=100)
    basemlp.fit(inputX, inputY)

    # Validate decision tree. But we aren't calibrating any parameters so this is optional.
    validX = validation_data.iloc[:, 0:1024]
    validY = validation_data.iloc[:,1024]
    score = basemlp.score(validX, validY)
    print(f'Score on validation set: {score}')

    # Predict using the test data
    testX = test_data.iloc[:,0:1024]
    testY = test_data.iloc[:,1024]
    predictions = basemlp.predict(testX)
    score2 = basemlp.score(testX, testY)
    
    # Print the predictions to CSV
    output_df = pd.DataFrame(predictions)
    output_df.transpose()
    fileutils.write_output(filename, output_df)

    return predictions, testY


print("===Base-MLP Data Set 1===")
input1 = fileutils.load_csv("train_1")
valid1 = fileutils.load_csv("val_1")
test1 = fileutils.load_csv("test_with_label_1")

test_predictions1, test_correct1 = predict_base_mlp(input1, valid1,  test1, "Base-MLP-DS1")

confusion_matrix.create(test_predictions1, test_correct1, alphabet.latin(), "Base-MLP-DS1")
metrics.compute(test_predictions1, test_correct1, alphabet.latin(), "Base-MLP-DS1")

print("===Base-MLP Data Set 2===")
input2 = fileutils.load_csv("train_2")
valid2 = fileutils.load_csv("val_2")
test2= fileutils.load_csv("test_with_label_2")

test_predictions2, test_correct2 = predict_base_mlp(input2, valid2, test2, "Base-MLP-DS2")

confusion_matrix.create(test_predictions2, test_correct2, alphabet.greek(), "Base-MLP-DS2")
metrics.compute(test_predictions2, test_correct2, alphabet.greek_no_unicode(), "Base-MLP-DS2")