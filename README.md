Loan Default Risk & Credit Rating Prediction

This project focuses on loan default risk classification and credit-style rating prediction using Machine Learning. The goal is to classify loans as Good (0) or Bad / Default Risk (1) and assign them into risk rating buckets (AAA → CCC) to help identify high-risk borrowers early.

The model is built using Logistic Regression with balanced class weights for strong interpretability and efficient training. The dataset contains both numerical and categorical features, with preprocessing performed using:

Numeric Features: Median Imputation + Standard Scaling
Categorical Features: Most Frequent Imputation + One-Hot Encoding

To address scalability and runtime optimization, the project applies Gaussian Random Projection for dimensionality reduction and trains on a 40% stratified subset to simulate a reduced dataset scenario.

Key Results

Baseline Model (Full Dataset)

Accuracy: 0.839
F1 Score: 0.685
ROC-AUC: 0.867
Train Time: 3.45 s

Random Projection + 40% Dataset

Accuracy: 0.835
F1 Score: 0.673
ROC-AUC: 0.862
Train Time: 1.52 s (~2.3× faster)

The optimized approach achieves significant training speed improvement with only a minimal performance drop.

Project Outputs
Model evaluation metrics (Accuracy, F1, ROC-AUC, Train/Test Time)
Risk rating buckets (AAA → CCC) generated from predicted default probabilities
CSV exports containing per-loan predictions and ratings
Visualization reports for metrics, runtime comparison, probability distributions, and rating distributions
Future Improvements
Gradient Boosting / HistGradientBoosting models
Probability calibration (Platt / Isotonic methods)
Cost-sensitive threshold optimization
Explainability using SHAP and coefficient analysis

This project demonstrates an end-to-end workflow for credit risk assessment, model optimization, dimensionality reduction, and business-oriented risk rating generation.
