---
author: Me
title: Credit Risk Analysis Project
date: 2025-09-13
description: Showcase of my Credit Risk Analysis project using Python, XGBoost, and TensorFlow.
tags:
 - credit risk
 - machine learning
 - xgboost
 - python
 - tensorflow
---

This article showcases my **Credit Risk Analysis project**, including methodology, code snippets, visualizations, and insights. The project demonstrates how to predict credit default using machine learning techniques and interpret the results effectively.
<!--more-->

## Project Overview

**Credit Risk Analysis** [Python, XGBoost, TensorFlow]  
- Developed an XGBoost model to predict credit default using the American Express dataset.  
- Achieved **92% prediction accuracy**, reducing potential default exposure by 18%.  
- Conducted **data preprocessing, hyperparameter tuning, bias/variance analysis**, and **SHAP analysis**.  
- Evaluated and demonstrated superior performance and interpretability compared to neural networks and ensemble models.  

For more details and full code, visit the project repository: [Credit Risk GitHub](https://github.com/pratzzeee/creditrisk)

## Data Exploration

- Dataset: American Express credit data  
- Key features: `age`, `income`, `credit_limit`, `payment_history`, `default_flag`  
- Missing value handling, outlier detection, and feature scaling were performed.  

<!-- ## Visualizations

![Credit Risk Distribution](https://picsum.photos/480/320)

> Data visualization helped identify trends and correlations between features and the target variable. -->

## Key Insights

- High correlation between **payment history** and **default risk**.  
- Certain income and credit limit brackets are more likely to default.  
- Model interpretability using **SHAP** highlighted key drivers of credit risk.

## Sample Code Snippets

#### XGBoost Model Training

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")