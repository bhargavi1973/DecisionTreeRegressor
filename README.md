Decision Tree Regressor using Breast Cancer Dataset
This repository contains a machine learning implementation using Decision Tree Regressor from the scikit-learn library. The project demonstrates how to apply regression algorithms to a classification dataset—Breast Cancer—for educational and exploratory purposes.

📌 Project Overview
Dataset: Breast Cancer Wisconsin Dataset (loaded via sklearn.datasets)
Objective: Predict binary class (malignant or benign) using a regression model and evaluate using classification metrics.
ML Model: DecisionTreeRegressor from sklearn.tree

📂 Workflow Summary
Load the Breast Cancer dataset using load_breast_cancer().
Prepare feature matrix X and target variable y.
Split data into training and testing sets.
Train a DecisionTreeRegressor.
Generate predictions and evaluate with metrics like:
Mean Squared Error (MSE)
Accuracy Score
Confusion Matrix
Recall Score
Classification Report

⚠️ Note: Although DecisionTreeRegressor is typically used for continuous outputs, it is applied here for a binary classification target—highlighting how regression outputs can be interpreted for classification using thresholding.

📊 Evaluation Metrics Used
mean_squared_error
accuracy_score (post-rounding or thresholding of predictions)
confusion_matrix
recall_scor
classification_report

📈 Output
The notebook prints the model parameters, performance scores, and a full classification report based on the regressor’s predictions.

🔍 Observations
The use of regression for binary classification is unconventional but insightful for experimentation.
Evaluation using classification metrics on regression output may lead to performance biases if not post-processed carefully.

🤝 Contribution
Contributions are welcome for:
Improving the use of classification-specific models.
Adding visualization (e.g., decision tree plot).
Hyperparameter tuning and cross-validation.

