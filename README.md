# KNN Classification on BEED Dataset

A MATLAB-based machine learning project for classifying samples from the **BEED dataset** using the **K-Nearest Neighbors (KNN)** algorithm.

The project includes data preprocessing, model training, prediction, performance evaluation, ROC curve analysis, AUC calculation, and 5-fold cross-validation.

---

## Overview

This project implements a **K-Nearest Neighbors (KNN) classifier** to classify samples from the BEED dataset.

The complete workflow is:

**Dataset → Preprocessing → Train/Test Split → KNN Classification → Performance Evaluation → ROC/AUC → Cross-Validation**

The objective is to evaluate how effectively KNN can distinguish between the classes present in the BEED dataset.

---

## Algorithm

### K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm that classifies a new sample based on the classes of its nearest neighboring training samples.

In this implementation:

* **Number of neighbors (K):** 5
* **Distance metric:** Euclidean
* **Training/Test split:** 80% / 20%
* **Cross-validation:** 5-fold
* **Feature standardization:** Enabled

For each test sample, the algorithm identifies the five closest training samples and determines the predicted class based on their labels.

---

## Dataset

The project uses:

```text
BEED_data.csv
```

The dataset is expected to contain:

* Feature columns
* One final column containing the class label

The MATLAB code separates the dataset into:

```text
X → Features
Y → Class Labels
```

---

## Workflow

### 1. Load Dataset

The dataset is imported using MATLAB's `readtable()` function.

```matlab
data = readtable('BEED_data.csv');
```

### 2. Extract Features and Labels

```matlab
X = table2array(data(:,1:end-1));
Y = data{:,end};
```

### 3. Feature Standardization

Features are transformed into standardized values using their mean and standard deviation.

```matlab
standardized_X = (X - mean_X) ./ std_X;
```

The project also visualizes the data before and after standardization.

### 4. Data Normalization

Features are normalized before classification:

```matlab
X = normalize(X);
```

This helps prevent features with larger numerical ranges from disproportionately influencing distance calculations.

### 5. Train-Test Split

The dataset is divided into:

* 80% training data
* 20% testing data

```matlab
cv = cvpartition(Y,'HoldOut',0.20);
```

### 6. Train KNN Classifier

A KNN model is trained using five nearest neighbors:

```matlab
k = 5;

knnModel = fitcknn(...
    XTrain,...
    YTrain,...
    'NumNeighbors',k,...
    'Distance','euclidean',...
    'Standardize',true);
```

### 7. Prediction

The trained model predicts the classes of unseen test samples:

```matlab
[YPred,scores] = predict(knnModel,XTest);
```

---

## Performance Evaluation

The classifier is evaluated using multiple performance metrics.

### Accuracy

Measures the percentage of correctly classified test samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Confusion Matrix

A confusion matrix is generated to visualize classification performance.

```matlab
confusionchart(YTest,YPred);
```

The matrix provides information about:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Precision

```text
Precision = TP / (TP + FP)
```

Precision measures how many samples predicted as positive actually belong to the positive class.

### Recall / Sensitivity

```text
Recall = TP / (TP + FN)
```

Recall measures how many actual positive samples were correctly detected.

### Specificity

```text
Specificity = TN / (TN + FP)
```

Specificity measures how effectively negative samples are correctly identified.

### F1 Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The F1 score provides a balance between precision and recall.

---

## ROC Curve and AUC

The project generates a **Receiver Operating Characteristic (ROC) curve** using MATLAB's `perfcurve()` function.

The ROC curve evaluates the classifier's ability to distinguish between classes at different classification thresholds.

The **Area Under the Curve (AUC)** is also calculated for each class.

```matlab
[Xroc,Yroc,~,AUC(i)] = perfcurve(...
    YTest,...
    scores(:,i),...
    classes(i));
```

A larger AUC generally indicates better class discrimination.

---

## 5-Fold Cross-Validation

To provide an additional estimate of model performance, 5-fold cross-validation is performed.

```matlab
CVModel = crossval(knnModel,'KFold',5);

cvLoss = kfoldLoss(CVModel);

cvAccuracy = (1-cvLoss)*100;
```

The dataset is divided into five subsets. Each subset is used as the validation set once, while the remaining subsets are used for training.

---

## Repository Structure

```text
BEED-KNN-Classification/
│
├── BEED_data.csv
├── KNN_Classification.m
├── README.md
│
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── standardization_plot.png
```

---

## Requirements

* MATLAB
* Statistics and Machine Learning Toolbox
* BEED dataset (`BEED_data.csv`)

The code uses MATLAB functions including:

```text
readtable()
cvpartition()
fitcknn()
predict()
confusionmat()
confusionchart()
perfcurve()
crossval()
kfoldLoss()
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/BEED-KNN-Classification.git
```

### 2. Open MATLAB

Navigate to the project directory.

### 3. Add the Dataset

Place:

```text
BEED_data.csv
```

in the same directory as the MATLAB script.

### 4. Run the Script

Open:

```text
KNN_Classification.m
```

and click **Run**.

The script will:

1. Load the dataset
2. Display the dataset summary
3. Preprocess the features
4. Split the data
5. Train the KNN classifier
6. Generate predictions
7. Calculate performance metrics
8. Display the confusion matrix
9. Generate ROC curves
10. Calculate AUC
11. Perform 5-fold cross-validation

---

## Results

The MATLAB implementation generates:

* Classification accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1 Score
* Confusion matrix
* ROC curve
* AUC for each class
* 5-fold cross-validation accuracy

Numerical results depend on the dataset and train-test partition generated during execution.

---

## Applications

KNN-based classification can be useful in biomedical machine learning applications including:

* Biomedical signal classification
* EEG classification
* Physiological signal analysis
* Disease classification
* Pattern recognition
* Clinical decision-support research

---

## Notes

The current implementation performs normalization before the train-test split and also enables standardization within `fitcknn`.

For rigorous research or publication, preprocessing should ideally be performed using parameters calculated **only from the training set** to prevent potential data leakage.

---

## Future Improvements

Potential improvements include:

* Hyperparameter optimization for K
* Comparison with SVM, Random Forest, LDA, and other classifiers
* Stratified cross-validation
* Feature selection
* Hyperparameter tuning
* Additional evaluation metrics
* Learning curves
* Statistical comparison between models
* Prevention of preprocessing data leakage
* Automated result reporting

---

## Author

**Ayesha Kauser**

BS Biomedical Engineering
University of Engineering and Technology, Lahore

---

## License

This project is intended for academic and research purposes.
