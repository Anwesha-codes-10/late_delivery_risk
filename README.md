# Late Delivery Prediction System

A machine learning project to predict whether an order 
will be delivered late using real supply chain transaction 
data from DataCo Global.

## Problem Statement
Late deliveries directly impact customer satisfaction 
and business revenue. This project builds a model that 
predicts delivery delays at the time of order placement, 
using only information available before shipment.

## Dataset
- Source: DataCo Supply Chain Dataset
- Repository: Mendeley Data
- Authors: Fabian Constante, Fernando Silva, António Pereira
- License: Creative Commons 4.0
- Size: 180,519 records, 53 features
- Target: Late_delivery_risk (1 = Late, 0 = On Time)

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Pickle

## Project Structure
- Data Loading & Exploration
- Data Cleaning & Leakage Detection
- Exploratory Data Analysis (EDA)
- Label Encoding
- Model Training (Logistic Regression, Random Forest, XGBoost)
- 5-Fold Cross Validation
- Hyperparameter Tuning with GridSearchCV
- Model Evaluation (Accuracy, F1, Recall, Confusion Matrix)
- Feature Importance Analysis
- Predictive System for single order prediction

## Results

| Model | Train Accuracy | Test Accuracy | Overfitting Gap |
|---|---|---|---|
| Logistic Regression | 68% | 68% | 0% |
| Random Forest | 99.6% | 68% | 31.6% |
| XGBoost (Tuned) | 73% | 71% | 2% |

- XGBoost selected as final model with 71% accuracy
- Random Forest showed severe overfitting — 31.6% gap
- Best Parameters: learning_rate=0.05, max_depth=5, 
  n_estimators=100

## Key Learnings
- Detected and removed data leakage columns that caused
  artificial 100% accuracy (Delivery Status, Days for 
  shipping real)
- Diagnosed severe overfitting in Random Forest
  vs minimal overfitting in XGBoost
- Dataset had 65,000+ nulls in Product Description — 
  dropped column instead of rows to preserve data
- Built end to end predictive system for new orders

## How to Run
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Download dataset from Mendeley Data repository
4. Open Late_Delivery_Prediction.ipynb in Jupyter or Colab
5. Run all cells top to bottom
