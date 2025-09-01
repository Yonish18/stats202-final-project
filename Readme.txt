This project was completed as the final assignment for STATS202: Data Mining and Statistical Learning at Stanford University (Summer 2025). The goal was to predict the relevance of a URL given a user query using a large dataset provided for a Kaggle competition.
The dataset included ~80,000 training rows and 30,000 test rows, with 10 different attributes describing query–URL pairs. Each row had a binary relevance label (1 = relevant, 0 = not relevant).
The task required building models that could preprocess the data, learn predictive relationships, and generate predictions for the Kaggle leaderboard.

Objectives:
Explore and visualize the dataset to understand attribute relationships.
Perform feature engineering and preprocessing where necessary.
Train and evaluate multiple models, focusing on classification performance.
Submit predictions to Kaggle and analyze leaderboard results.

Methods:
Preprocessing & Feature Engineering
Checked for missing values and attribute distributions.
Normalized and scaled numerical features where appropriate.
Created new features from existing attributes when correlations suggested potential improvements.

Models Implemented:
Logistic Regression (baseline): Provided a simple interpretable benchmark.
Random Forest Classifier: Tuned hyperparameters such as number of trees and max depth to improve accuracy.
Cross-validation was used to evaluate model performance before Kaggle submission.

Evaluation Metrics
Accuracy (primary Kaggle metric).
Precision, Recall, and F1-Score were also reported to understand class balance performance.

Results:
Random Forest achieved a validation accuracy of ~66% and performed best overall.
Logistic Regression served as a solid baseline but underperformed compared to ensemble methods.
Kaggle leaderboard score for the final Random Forest submission: 0.6577
.
Key Learnings
Feature understanding and visualization are as important as model choice in real-world data mining.
Simple models can provide strong baselines, but ensemble methods like Random Forest offer improved performance with structured data.
Working with large datasets requires careful organization of code and reproducible pipelines.

Tools & Libraries:
Python 3.10
pandas, numpy, matplotlib for data processing and visualization
scikit-learn for modeling (Logistic Regression, Random Forest)
Kaggle API for competition submission

Reproducibility:
Scripts included:
split_data.py — train/validation split.
step2_rf_baseline.py — Random Forest model training and evaluation.
logisticreg_model.py — Logistic Regression baseline.
check_accuracy.py — evaluation script with confusion matrix and classification report.

Author
Yonish Tayal
Boston University / Stanford University Summer Session (2025)
