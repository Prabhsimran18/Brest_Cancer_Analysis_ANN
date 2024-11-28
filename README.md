---

# Breast_Cancer_Analysis_ANN

This project involves analyzing the Breast Cancer Wisconsin Diagnostic dataset, implementing an Artificial Neural Network (ANN) for prediction, and deploying an interactive web application using Streamlit.

## Features

- **Data Loading and Preprocessing**
- **Feature Selection** using `SelectKBest`
- **Hyperparameter Tuning** with Grid Search CV
- **ANN Model Implementation** using `MLPClassifier`
- **Interactive Web App** for Predictions

## Results

### Missing Values in Each Feature

```plaintext
mean radius                0
mean texture               0
mean perimeter             0
mean area                  0
mean smoothness            0
mean compactness           0
mean concavity             0
mean concave points        0
mean symmetry              0
mean fractal dimension     0
radius error               0
texture error              0
perimeter error            0
area error                 0
smoothness error           0
compactness error          0
concavity error            0
concave points error       0
symmetry error             0
fractal dimension error    0
worst radius               0
worst texture              0
worst perimeter            0
worst area                 0
worst smoothness           0
worst compactness          0
worst concavity            0
worst concave points       0
worst symmetry             0
worst fractal dimension    0
dtype: int64
```

### Selected Features

```python
['mean radius', 'mean perimeter', 'mean area', 'mean concavity', 'mean concave points', 
 'worst radius', 'worst perimeter', 'worst area', 'worst concavity', 'worst concave points']
```

### Grid Search CV Results

```
Fitting 5 folds for each of 48 candidates, totalling 240 fits
Best parameters found:
 {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (100,), 
  'learning_rate': 'constant', 'solver': 'adam'}
```

### Confusion Matrix

```
[[42  1]
 [ 2 69]]
```

### Classification Report

```
              precision    recall  f1-score   support

          0       0.95      0.98      0.97        43
          1       0.99      0.97      0.98        71

   accuracy                           0.97       114
  macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

## Model Performance

- **Accuracy:** 97%
- **Precision:** 95% (Malignant), 99% (Benign)
- **Recall:** 98% (Malignant), 97% (Benign)
- **F1-Score:** 0.97 (Malignant), 0.98 (Benign)

The model demonstrates high accuracy and excellent precision and recall, indicating reliable performance in distinguishing between malignant and benign tumors.
