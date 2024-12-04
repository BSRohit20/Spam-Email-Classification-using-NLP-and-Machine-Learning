
# Spam Email Classification using NLP and ML

This project is a simple spam email classification system using Naive Bayes and Count Vectorizer in Python. It reads a dataset of labeled spam and ham (non-spam) emails, trains a Naive Bayes classifier on the data, and provides a function to predict whether a new email is spam or not.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Training the Model](#training-the-model)
- [Classifying Emails](#classifying-emails)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Usage](#usage)

## Overview
The project uses a dataset of emails, with labels for spam and ham messages. The goal is to predict whether a given email is spam or ham using a machine learning model. The project implements a simple pipeline for:
- Data cleaning
- Feature extraction (using CountVectorizer)
- Model training (Naive Bayes)
- Saving and loading the trained model using `pickle`
- Classifying new emails

## Prerequisites
To run this project, you need to have Python 3.x and the following libraries installed:

- pandas
- numpy
- scikit-learn
- pickle (standard library, no need to install separately)

You can install the required libraries using `pip`:

```bash
pip install pandas numpy scikit-learn
```

## Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spam-email-classification.git
   cd spam-email-classification
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset `spam.csv` from your source and place it in the project directory.

## Dataset
This project uses the `spam.csv` dataset, which contains email messages and their corresponding labels (spam or ham). The dataset is expected to have the following columns:
- `v1`: Class of the email (ham or spam)
- `v2`: Message content of the email

The columns `Unnamed: 2`, `Unnamed: 3`, and `Unnamed: 4` are dropped since they don't contain useful information.

## Code Structure
The code can be divided into the following sections:
1. **Data Preprocessing**:
   - Loads the dataset and cleans unnecessary columns.
   - Converts labels from strings (`ham`, `spam`) to numerical values (0 for ham, 1 for spam).
   - Handles missing values (if any).

2. **Feature Extraction**:
   - Uses `CountVectorizer` to convert the email text into numerical features suitable for model training.

3. **Model Training**:
   - Trains a Naive Bayes classifier using the processed data.

4. **Model Evaluation**:
   - Evaluates the model's performance on a test set.

5. **Saving the Model**:
   - Saves the trained model and `CountVectorizer` using `pickle` for later use.

6. **Email Classification**:
   - Classifies new email messages as either spam or ham.

## Training the Model
To train the model, the following steps are performed:

1. Load the dataset.
2. Preprocess the data (remove unnecessary columns and handle missing values).
3. Convert text data into numerical features using `CountVectorizer`.
4. Split the data into training and test sets using `train_test_split`.
5. Train the `MultinomialNB` Naive Bayes classifier on the training data.
6. Evaluate the model's accuracy on the test data.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('spam.csv', encoding='latin-1')

# Preprocess data
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data['class'] = data['class'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['class']

# Convert text data to numerical data using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy}')
```

## Classifying Emails
To classify new emails, the model and `CountVectorizer` are loaded from `pickle` files. The email text is transformed into numerical features using the same `CountVectorizer`, and the model predicts whether the email is spam or ham.

```python
import pickle

# Load the trained model and CountVectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vec.pkl', 'rb'))

def classify_email(email_text):
    vect = cv.transform([email_text]).toarray()
    prediction = model.predict(vect)
    
    return "spam" if prediction == 1 else "ham"

# Example usage
email = input("Enter the email message: ")
classification = classify_email(email)
print(f"The email is classified as: {classification}")
```

## Saving and Loading the Model
The trained model and `CountVectorizer` are saved using `pickle`:

```python
# Save the model
pickle.dump(model, open('spam.pkl', 'wb'))
pickle.dump(cv, open('vec.pkl', 'wb'))
```

These files (`spam.pkl` and `vec.pkl`) can be loaded later for making predictions on new data.

## Usage
1. **Train the model** by running the provided code and saving the model and vectorizer.
2. **Classify new emails** using the `classify_email` function after loading the saved model and vectorizer.

```python
email = input("Enter the email message: ")
classification = classify_email(email)
print(f"The email is classified as: {classification}")
```



Feel free to adjust the content to suit your exact project structure, particularly the setup and usage instructions.
