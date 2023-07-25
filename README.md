# Predicting User Purchase Intent with Nearest-Neighbor Classifier

## Background

When users shop online, not all of them end up making a purchase. Predicting whether a user intends to make a purchase or not can be valuable for online shopping websites. By understanding the user's intent, the website can tailor the user's experience, such as showing discounts to users who are unlikely to complete the purchase. Machine learning comes into play to solve this problem.

The task is to build a nearest-neighbor classifier to predict whether a user will make a purchase or not based on information about the user's browsing session. This classifier should be better than random guessing and should aim to achieve a reasonable balance between sensitivity (true positive rate) and specificity (true negative rate).

## Understanding the Data

The dataset provided in `shopping.csv` contains information from approximately 12,000 user sessions. Each row represents one user session, and the columns capture various aspects of the user's behavior during the session. The data includes information about the number of pages visited, session duration, bounce rates, exit rates, page values, special day proximity, user agent details, operating systems, browsers, regions, traffic type, visitor type (returning or non-returning), and whether the session occurred on a weekend. The most critical column is the `Revenue` column, which indicates whether the user made a purchase (TRUE) or not (FALSE).

## Functions Implemented

The following functions have been implemented in `shopping.py` to build and evaluate the nearest-neighbor classifier:

### `load_data(filename)`

This function loads data from a CSV file and returns a tuple `(evidence, labels)`. The `evidence` list contains the feature vectors for each data point, and the `labels` list contains the corresponding labels indicating whether a purchase was made or not.

### `train_model(evidence, labels)`

This function takes `evidence` and `labels` as input and returns a trained scikit-learn nearest-neighbor classifier (KNeighborsClassifier with k = 1).

### `evaluate(true_labels, predicted_labels)`

This function evaluates the classifier's performance by calculating sensitivity and specificity based on the true labels and predicted labels.

## Usage

To use the nearest-neighbor classifier, follow these steps:

1. Import the necessary libraries and functions from `shopping.py`.
2. Load the data from the CSV file using the `load_data` function.
3. Split the data into a training set and a testing set.
4. Train the nearest-neighbor classifier using the `train_model` function on the training data.
5. Make predictions on the testing data using the trained model.
6. Evaluate the classifier's performance using the `evaluate` function, which will provide sensitivity and specificity metrics.

## Data Preprocessing

Before training the nearest-neighbor classifier, ensure that all data points are in numeric format. The data types should be as follows:

- `Administrative`, `Informational`, `ProductRelated`, `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, and `Weekend` should be of type `int`.
- `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, and `SpecialDay` should be of type `float`.
- Convert the `Month` column to numeric values: 0 for January, 1 for February, and so on up to 11 for December.
- Convert the `VisitorType` column to 1 for returning visitors and 0 for non-returning visitors.
- Convert the `Weekend` column to 1 if the user visited on a weekend and 0 otherwise.

## Evaluation Metrics

The classifier's performance will be evaluated based on sensitivity and specificity metrics:

- Sensitivity: Represents the true positive rate, which measures the proportion of actual positive labels (users who made a purchase) that were correctly identified.
- Specificity: Represents the true negative rate, which measures the proportion of actual negative labels (users who did not make a purchase) that were correctly identified.
