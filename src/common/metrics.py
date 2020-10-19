from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
import csv;
import os;

dirname = os.path.dirname(__file__)

def print_metric(writer, classes, metrics):
    for i in range(len(classes)):
        writer.writerow([classes[i], metrics[i]])

def compute(test_predictions, test_correct, alphabet, filename):
    metrics = precision_recall_fscore_support(test_correct, test_predictions, average=None, zero_division=0)
    accuracy = accuracy_score(test_correct, test_predictions)
    f1_macro = f1_score(test_correct, test_predictions, average='macro')
    f1_weighted = f1_score(test_correct, test_predictions, average='weighted')

    with open(dirname + f'/../../output/{filename}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(['Precision'])
        print_metric(writer, alphabet, metrics[0])
        writer.writerow([])
        writer.writerow(['Recall'])
        print_metric(writer, alphabet, metrics[1])
        writer.writerow([])
        writer.writerow(['F1-Measure'])
        print_metric(writer, alphabet, metrics[2])
        writer.writerow([])
        writer.writerow(['Occurence Count'])
        print_metric(writer, alphabet, metrics[3])
        writer.writerow([])
        writer.writerow(['Accuracy', accuracy])
        writer.writerow(['Macro F1-Measure', f1_macro])
        writer.writerow(['Weighted F1-Measure', f1_weighted])
