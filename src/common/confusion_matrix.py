from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import os
import csv
dirname = os.path.dirname(__file__)

def create(test_predictions, test_correct, headers, filename):
    confusion = confusion_matrix(test_correct, test_predictions)

    # Print the confusion matrix
    with open(dirname + f'/../../output/{filename}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Confusion Matrix"])
        writer.writerows(confusion)

            


    # Count occurrences to fill out values of the matrix
    headers.extend(["="])
    headers_length = len(headers)
    alphabet_length = headers_length - 1
    
    values = [[0 for x in range(headers_length)] for y in range(headers_length)]

    for i in range(len(confusion)):
        for j in range(len(confusion[i])):
            #Transpose here so that we have correct as the x axis and predictions as the y axis
            values[i][j] = confusion[j][i]

    # Sum up totals for rows and columns
    total = 0
    for i in range(alphabet_length):
        sum_horizontal = 0
        sum_vertical = 0
        for j in range(alphabet_length):
            sum_horizontal += values[i][j]
            sum_vertical += values[j][i]
        values[i][alphabet_length] = sum_horizontal
        values[alphabet_length][i] = sum_vertical
        total += sum_horizontal
    values[alphabet_length][alphabet_length] = total

    # Cell colors
    cellColours = [['#ffffff' for x in range(headers_length)] for y in range(headers_length)] # default to white
    for i in range(headers_length):
        for j in range(headers_length):
            if i == headers_length - 1 or j == headers_length - 1:
                cellColours[i][j] = '#929591' # grey for totals
            elif i == j and values[i][j] > 0:
                cellColours[i][j] = '#0165fc' # blue for correct predictions
            elif i == j and values[i][j] == 0:
                cellColours[i][j] = '#ff474c' # red for incorrect predictions
            elif values[i][j] > 0:
                cellColours[i][j] = '#ff474c' # red for incorrect predictions
            

    # Create and show the table
    table = pyplot.table(cellText=values, rowLabels = headers, colLabels=headers, loc='center', cellColours = cellColours)
    pyplot.axis('off')
    pyplot.show()