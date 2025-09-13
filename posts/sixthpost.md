---
author: Me
title: SMS Spam Classifier Project
date: 2025-09-13
description: A machine learning project to classify SMS messages as spam or ham using Python and scikit-learn.
tags:
 - spam detection
 - machine learning
 - text classification
 - python
 - NLP
---

This article showcases my **SMS Spam Classifier project**, which demonstrates how to preprocess text, engineer features, train models, and classify messages as spam or ham.
<!--more-->

## Project Overview

**Spam Classifier** [Python, Scikit-learn, NLTK]  
- Developed a machine learning model to classify SMS messages as spam or ham using the **SMS Spam Collection dataset**.  
- Achieved **>95% classification accuracy**, with high precision and recall.  
- Conducted **text preprocessing, feature engineering**, and **visualization** to improve model performance and interpretability.  

Project repository: [GitHub - SMS Spam Classifier](https://github.com/pratzzeee/sms-spam-classifier)  
Dataset: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Data Preprocessing

- **Lowercasing** all text  
- **Tokenization** using NLTK  
- **Stopword & punctuation removal**  
- **Stemming** with PorterStemmer  

## Feature Engineering

- Message length, word count, and sentence count  
- TF-IDF vectorization of processed text  

<!-- ## Visualization

![WordCloud](https://picsum.photos/480/320)  
- WordClouds for spam vs ham messages  
- Distribution plots for characters, words, and sentences  

> Visualizations helped identify distinguishing patterns between spam and ham messages. -->

## Model Training

- Models used: **Naive Bayes**, **Logistic Regression**, and other supervised learning algorithms  
- Metrics evaluated: **Accuracy, Precision, Recall, F1-score**

#### Example: Logistic Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))