# Credit Card Fraud Detection

This repository contains a Credit Card Fraud Detection project aimed at identifying fraudulent transactions from credit card data using machine learning (ML) models. The dataset used in this project is highly imbalanced, containing a small percentage of fraud cases. Various techniques such as SMOTEENN for balancing data and a range of classification models were implemented to address this challenge.

## Overview

Credit card fraud poses a significant challenge for businesses, banks and customers alike, making the detection of fraudulent transactions crucial. This project applies ML techniques to identify fraudulent transactions in a dataset containing both fraudulent and legitimate transactions.

---

## Dataset

The dataset used for this project is the **Credit Card Fraud Detection** dataset from Kaggle. It contains transactions made by European cardholders over two days in September 2013, consisting of 284,807 transactions, out of which 492 are fraudulent. The data is highly imbalanced, with fraud cases representing only 0.172% of all transactions.

- **Features**: Anonymized features (`V1` to `V28`), along with `Amount` and `Time`.
- **Target**: A binary label (`0` for legitimate, `1` for fraud).

Dataset link: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Preprocessing

Before building the models, the following preprocessing steps were performed:

- **Data Imbalance Handling**: Given the skewed nature of the data, SMOTEENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbors) was used to balance the dataset.
- **Scaling**: Features were scaled using StandardScaler.
- **Train-Test Split**: The dataset was split into training and testing sets, with 80% of the data used for training and 20% for testing.

---

## Modeling

Several ML algorithms were explored for detecting fraudulent transactions:

- **Random Forest Classifier (RF)**
- **Support Vector Machine (SVM)**
- **Logistic Regression (LR)**
- **Naive Bayes (NB)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree (DT)**

---

## Evaluation

Given the imbalanced nature of the dataset, a range of evaluation metrics were considered to provide a comprehensive assessment of model performance:

- **Accuracy**: Measures the overall correctness of the model's predictions.
- **Precision**: The proportion of predicted fraud cases that are actual frauds.
- **Recall (Sensitivity)**: The proportion of actual fraud cases correctly identified by the model.
- **Specificity**: The proportion of legitimate transactions correctly identified.
- **F1 Score**: The harmonic mean of precision and recall, balancing the two metrics.
- **Matthews Correlation Coefficient (MCC)**: A balanced measure that accounts for true and false positives and negatives, especially useful for imbalanced datasets.
- **ROC-AUC**: Provides a graphical representation of the model's ability to distinguish between classes, summarizing performance across various threshold values.

To assess the impact of data balancing, we evaluated our model's performance on both the unbalanced and balanced datasets. This two-stage evaluation allows for a comprehensive understanding of the model's behavior with unbalanced data and the effectiveness of the balancing techniques in improving performance.

---

## Results

The **Random Forest** model performed exceptionally well and was the best model for detecting fraudulent transactions. Below are the key performance metrics of the Random Forest model on the balanced dataset:

- **Accuracy**: 100.00%
- **Sensitivity (Recall)**: 1.00
- **Specificity**: 1.00
- **Matthews Correlation Coefficient (MCC)**: 1.00
- **Area Under the Curve (AUC)**: 1.00

The model achieved perfect scores on all key metrics, demonstrating its ability to correctly classify both legitimate and fraudulent transactions with 100% accuracy.
---

## Conclusion

This project demonstrates the effectiveness of ML models in detecting fraudulent transactions, with a particular focus on addressing data imbalance and outliers. The use of techniques like SMOTEENN can significantly improve model performance on imbalanced datasets.
