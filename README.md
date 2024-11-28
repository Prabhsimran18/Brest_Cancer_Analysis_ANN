# Brest_Cancer_Analysis_ANN
This project involves analyzing the Breast Cancer Wisconsin Diagnostic dataset, implementing an Artificial Neural Network (ANN) for prediction, and deploying an interactive web application using Streamlit.
# Features
* Data loading and preprocessing
* Feature selection using SelectKBest
* Hyperparameter tuning with Grid Search CV
* ANN model implementation using MLPClassifier
* Interactive web app for predictions

# Results
Present the evaluation metrics, confusion matrix, and any observations from the model's performance.
Missing values in each feature:
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
Selected Features: ['mean radius', 'mean perimeter', 'mean area', 'mean concavity', 'mean concave points', 'worst radius', 'worst perimeter', 'worst area', 'worst concavity', 'worst concave points']

Fitting 5 folds for each of 48 candidates, totalling 240 fits
Best parameters found:
 {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'solver': 'adam'}
Confusion Matrix:
 [[42  1]
 [ 2 69]]
Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.98      0.97        43
           1       0.99      0.97      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
