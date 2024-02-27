# CODSOFT-TASK-4
# Spam Detection using LightGBM
## Overview
This code demonstrates the implementation of a spam detection model using LightGBM, a gradient boosting framework. The dataset used for training and testing the model is named "spam.csv." The code involves data preprocessing, text cleaning, feature engineering, model training, and evaluation.

## Steps
## Step 1: Importing Libraries
The necessary libraries are imported, including pandas, numpy, matplotlib.pyplot, plotly_express, wordcloud, nltk, warnings, and the required modules for text preprocessing and machine learning.
## Step 2: Reading the Dataset
The code reads the dataset "spam.csv" into a Pandas DataFrame (df).
## Step 3: Data Visualization
Visualizes the class distribution using a histogram and a pie chart for the spam and ham (non-spam) messages.
## Step 4: Text Preprocessing
Performs several text preprocessing steps on the messages:
Replaces email addresses with 'emailaddress'.
Replaces URLs with 'webaddress'.
Replaces currency symbols with 'money-symbol'.
Replaces phone numbers with 'phone-number'.
Replaces numeric digits with 'number'.
Removes punctuation, extra whitespaces, and converts to lowercase.
Removes stop words using NLTK and applies stemming.
## Step 5: Feature Engineering
Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to represent text messages as numerical features.
## Step 6: Model Training and Testing
Splits the dataset into training and testing sets.
Uses LightGBM to train a spam detection model.
Evaluates the model using F1 score.
## Step 7: Hyperparameter Tuning
Uses RandomizedSearchCV to perform hyperparameter tuning for improving model performance.
## Step 8: Model Evaluation
Evaluates the model's performance on the test set using F1 score.
### Note
This code serves as a basic implementation for spam detection. Additional optimizations, parameter tuning, or model selection may be considered for further improvement.
