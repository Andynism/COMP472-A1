import os

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from common import confusion_matrix
from common import fileutils
from common import metrics

def best_mlp_params(input_data):
    inputX = input_data.iloc[:,0:1024] # Binary features (first 1024 columns)
    inputY = input_data.iloc[:,1024] # Index representing the class (last column)

    # Create a multilayered perceptron fit using the training data
    mlp = MLPClassifier()
    parameter_space = {'hidden_layer_sizes': [(20,15), (10,10,10)], 'activation': ['tanh', 'relu', 'logistic','identity'], 'solver': ['sgd', 'adam']}
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(inputX, inputY)

    # Best paramete set
    for i in clf.best_params_:
        print(i, clf.best_params_[i])
    
    return clf.best_params_

def predict_best_mlp(input_data, validation_data, test_data, filename):
    inputX = input_data.iloc[:,0:1024] # Binary features (first 1024 columns)
    inputY = input_data.iloc[:,1024] # Index representing the class (last column)

    # Create a decision tree fit using the training data
    basemlp = MLPClassifier(**best_mlp_params(input_data))
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

test_predictions1, test_correct1 = predict_best_mlp(input1, valid1,  test1, "Best-MLP-DS1")

confusion_matrix.create_alphabet(test_predictions1, test_correct1)
metrics.compute(test_predictions1, test_correct1, 26)

print("===Base-MLP Data Set 2===")
input2 = fileutils.load_csv("train_2")
valid2 = fileutils.load_csv("val_2")
test2= fileutils.load_csv("test_with_label_2")

test_predictions2, test_correct2 = predict_best_mlp(input2, valid2, test2, "Best-MLP-DS2")

confusion_matrix.create_greek(test_predictions2, test_correct2)
metrics.compute(test_predictions2, test_correct2, 10)