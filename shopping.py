import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer 0 
        - Administrative_Duration, a floating point number 1
        - Informational, an integer 2
        - Informational_Duration, a floating point number 3
        - ProductRelated, an integer 4
        - ProductRelated_Duration, a floating point number 5
        - BounceRates, a floating point number 6
        - ExitRates, a floating point number 7 
        - PageValues, a floating point number 8
        - SpecialDay, a floating point number 9
        - Month, an index from 0 (January) to 11 (December) 10
        - OperatingSystems, an integer 11 
        - Browser, an integer 12
        - Region, an integer 13
        - TrafficType, an integer 14
        - VisitorType, an integer 0 (not returning) or 1 (returning) 15
        - Weekend, an integer 0 (if false) or 1 (if true) 16

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    types = {0: int, 1: float, 2: int, 3: float, 4: int, 5: float, 6: float,
              7: float, 8: float, 9: float, 10: int, 11: int, 12: int, 13: int,
               14: int, 15: int, 16: int } 
    months = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
               'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    visitor = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 0}
    weekend = {'FALSE': 0, 'TRUE': 1}

    evidences = []
    labels = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) # skip the header row
        for row in csv_reader:
            labels.append(1 if row[-1] == 'TRUE' else 0)
            # Slice the first 17 columns and append to the list
            unprocessed_row = row[:17]
            unprocessed_row[10] = months[unprocessed_row[10]]
            unprocessed_row[15] = visitor[unprocessed_row[15]]
            unprocessed_row[16] = weekend[unprocessed_row[16]]
            single_evidence = []
            # ASSUMING order won't change
            for index, value in enumerate(unprocessed_row):
                if types[index] == int:
                    single_evidence.append(int(value))
                elif types[index] == float:
                    single_evidence.append(float(value))
                else:
                    raise Exception("Unexpected Error")
            
            evidences.append(single_evidence)
            
    return (evidences, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knn = KNeighborsClassifier(n_neighbors=1)  
    knn.fit(evidence, labels)  
    return knn


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    assert len(labels) == len(predictions)
    true_positive_count = 0
    true_pred_positive_count = 0
    true_negative_count = 0
    true_pred_negative_count = 0
    for ypred, y in zip(predictions, labels):
        if y == 1:
            true_positive_count += 1
            if ypred == 1:
                true_pred_positive_count +=1 
        elif y == 0:
            true_negative_count +=1 
            if ypred == 0:
                true_pred_negative_count +=1 
        else:
            raise Exception("Unexpected label")

    sensitivity = true_pred_positive_count/true_positive_count
    specificity = true_pred_negative_count/true_negative_count       
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
