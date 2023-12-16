# Breast Cancer Classification Project 🩺

## Overview
This project aims to classify breast cancer using Decision Tree and Support Vector Machine (SVM) models.

## 📚 Libraries Used
- NumPy
- Matplotlib
- scikit-learn

## 📊 Data
The breast cancer dataset is loaded using scikit-learn's `load_breast_cancer` function.

## 🛠️ Model Training
Two models are trained: Decision Tree and Support Vector Machine (SVM).

## 📈 Evaluation
Model performance is evaluated using confusion matrix and classification report.

## 📌 Feature Importance
Decision Tree feature importance is analyzed and displayed.

## 🚀 Hyperparameter Tuning
GridSearchCV is used for hyperparameter tuning of the Decision Tree model.

## 🏁 Final Model
The final Decision Tree model is trained with optimized hyperparameters.

## 📈 SVM Classification
Additionally, an SVM model is trained using a linear kernel.

## 📊 Visualization
Decision Tree, SVM, and final model visualizations are provided.

📊 Data Set Characteristics:
Number of Instances: 569
Number of Attributes: 30 numeric, predictive attributes, and the class
Attributes:
Radius (mean of distances from center to points on the perimeter)
Texture (standard deviation of gray-scale values)
Perimeter
Area
Smoothness (local variation in radius lengths)
Compactness (perimeter^2 / area - 1.0)
Concavity (severity of concave portions of the contour)
Concave Points (number of concave portions of the contour)
Symmetry
Fractal Dimension ("coastline approximation" - 1)
Class:
WDBC-Malignant
WDBC-Benign
Summary Statistics:
Min to Max ranges for various attributes.
Dataset Details:
This dataset consists of 569 instances with 30 numeric features describing characteristics of cell nuclei in breast mass images. The class distribution includes 212 Malignant and 357 Benign cases.

References:
W.N. Street, W.H. Wolberg, and O.L. Mangasarian. "Nuclear feature extraction for breast tumor diagnosis." IS&T/SPIE 1993 International Symposium on Electronic Imaging.
O.L. Mangasarian, W.N. Street, and W.H. Wolberg. "Breast cancer diagnosis and prognosis via linear programming." Operations Research, 43(4), July-August 1995.
W.H. Wolberg, W.N. Street, and O.L. Mangasarian. "Machine learning techniques to diagnose breast cancer from fine-needle aspirates." Cancer Letters 77 (1994) 163-171.
Model Performance:
Decision Tree (DT):
Train Score: 1.0
Test Score: 0.9532
Confusion Matrix:
[[ 57   5]
 [  3 106]]
Classification Report:
              precision    recall  f1-score   support
malignant    0.95000000 0.91935484 0.93442623 62
benign       0.95495495 0.97247706 0.96363636 109
accuracy                              0.95321637 171
Feature Importance:
Decision Tree Feature Importance:

Mean Radius: 0.0071
Mean Texture: 0.0357
Mean Compactness: 0.0071
Mean Concavity: 0.0158
Mean Concave Points: 0.0163
Area Error: 0.0139
Worst Texture: 0.0552
Worst Area: 0.7191
Worst Concave Points: 0.1297
Decision Tree Visualization:
A snippet of the Decision Tree is provided for better interpretability. Notably, the tree reveals key features influencing the classification.

|--- worst area <= 874.85
|   |--- worst concave points <= 0.16
|   |   |--- mean concave points <= 0.05
|   |   |   |--- ... (truncated for brevity)
|   |--- worst area >  874.85
|   |   |--- mean concavity >  0.07
|   |   |   |--- class: 0

SVM Classification:
The SVM model, trained with a linear kernel, achieves an accuracy of [insert accuracy here]. For detailed evaluation metrics, refer to the classification report.

Hyperparameter Tuning:
Grid Search parameters for Decision Tree:

Criterion: 'gini'
Max Depth: 2
After hyperparameter tuning, the final Decision Tree model achieved:

Train Score: 0.9598
Test Score: 0.9357
Conclusion:
The project successfully classifies breast cancer using the provided dataset. The Decision Tree model demonstrates strong performance, and hyperparameter tuning further refines the model. The feature importance analysis provides insights into the crucial factors influencing the classification decision.

For more details, refer to the complete code.

Feel free to explore the Decision Tree visualization for an in-depth understanding of the classification process.
