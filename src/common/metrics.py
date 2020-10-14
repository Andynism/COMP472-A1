def compute(test_predictions, test_correct, alphabet_size):
    # Count true positives, false positives, false negatives
    true_positives = [0 for x in range(alphabet_size)]
    false_positives = [0 for x in range(alphabet_size)]
    false_negatives = [0 for x in range(alphabet_size)]
    class_count = [0 for x in range(alphabet_size)]

    count_correct = 0
    count_incorrect = 0

    for i in range(len(test_predictions)):
        prediction = test_predictions[i]
        correct = test_correct[i]

        class_count[correct] += 1 

        if(correct == prediction):
            count_correct += 1
            true_positives[prediction] += 1
        else:
            count_incorrect += 1
            false_positives[prediction] += 1
            false_negatives[correct] += 1



    # Precision, Recall, F1 measure for each class
    precision = [0 for x in range(alphabet_size)]
    recall = [0 for x in range(alphabet_size)]
    f1_measure = [0 for x in range(alphabet_size)]

    for i in range(alphabet_size):
        if(true_positives[i] + false_positives[i] != 0):
            precision[i] = true_positives[i] / (true_positives[i] + false_positives[i])
        
        if(true_positives[i] + false_negatives[i] != 0):
            recall[i] = true_positives[i] / (true_positives[i] + false_negatives[i])
        
        if(precision[i] + recall[i] != 0):
            f1_measure[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    # Accuracy, macro-average F1 and weighted-average F1 of the whole model
    total_instances = count_correct + count_incorrect
    accuracy = count_correct / total_instances


    weighted_average_f1 = 0
    macro_average_f1 = 0
    
    for i in range(alphabet_size):
        macro_average_f1 += f1_measure[i]
        weighted_average_f1 += f1_measure[i] * class_count[i] / total_instances

    macro_average_f1 /= alphabet_size

    print(f'Accuracy: {accuracy}')
    print(f'Macro average f1: {macro_average_f1}')
    print(f'Weighted average f1: {weighted_average_f1}')
