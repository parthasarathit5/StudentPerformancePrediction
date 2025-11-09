# Student Performance Prediction using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
# Example dataset: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
df = pd.read_csv("StudentsPerformance.csv")

# Encode categorical columns
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
df['parental level of education'] = le.fit_transform(df['parental level of education'])
df['lunch'] = le.fit_transform(df['lunch'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])

# Define features and target
X = df[['gender', 'race/ethnicity', 'lunch', 'test preparation course', 
        'math score', 'reading score']]
df['pass'] = df['writing score'].apply(lambda x: 1 if x >= 50 else 0)  # Pass/Fail target
y = df['pass']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Accuracy=TP+TN/FP+FN+TP+TN​
# | Metric        | Formula               | Measures                           | When to Use                           |
# | ------------- | --------------------- | ---------------------------------- | ------------------------------------- |
# | **Precision** | TP / (TP + FP)        | Accuracy of positive predictions   | When false positives are costly       |
# | **Recall**    | TP / (TP + FN)        | Coverage of actual positives       | When false negatives are costly       |
# | **F1-Score**  | 2 × (P × R) / (P + R) | Balance between precision & recall | When you need both balance & fairness |
# | **Accuracy**  | (TP + TN) / Total     | Overall correctness                | Only when classes are balanced        |



# “What is the output?”

# 👉 Best Answer:

# The output of my model is a prediction — it predicts whether a student will Pass (1) or Fail (0) 
# based on their input features.
# For example, if a student studied for 7 hours with 85% attendance, the model predicts
# a high chance of passing.

# My model achieved around 96% accuracy, which means it correctly predicted 192 out of 200 students.
# It was especially strong in predicting pass students (98% correct),
# and fairly good at predicting fail students (84% correct).

# Simple HR-friendly Answer:

# My project predicts whether a student will pass or fail based on their performance data.
# I used machine learning to find patterns in how study hours, attendance, and scores affect results.
# The goal is to help teachers identify weak students early so they can guide them better.
# The model predicts results with about 96% accuracy.

# ✅ HR Tip: Speak slowly, clearly, and smile — HR doesn’t test your coding,
# they test your clarity and confidence.

# | Question              | Short Answer                              |
# | --------------------- | ----------------------------------------- |
# | What is your project? | Predicting student pass/fail using ML     |
# | Why this project?     | Solves real-world academic problem        |
# | How did you do it?    | Data → Clean → Train model → Predict      |
# | Output?               | Pass/Fail prediction, 96% accuracy        |
# | What tools?           | Python, Pandas, Sklearn, Matplotlib       |
# | What you learned?     | End-to-end ML process, evaluation metrics |


# 1️⃣ HR Round – Simple, Easy-to-Understand Answer

# 👉 Say this naturally and clearly:

# Precision means how accurate my model is when it predicts something as “positive”.
# For example, in my project, if the model predicts a student will pass, precision tells me how many of those predictions were actually correct.

# Recall means how many of the actual pass students my model was able to find correctly.
# It shows how well the model is catching all real cases.

# F1-score is a balance between precision and recall — it gives an overall measure of model performance.

# So in short:

# Precision → correctness of predictions

# Recall → completeness of predictions

# F1-score → balanced accuracy

# ✅ Example (say this if HR asks for one):

# My model’s precision was 0.98, which means it correctly identified 98% of pass students.
# Its recall was 0.98 too, meaning it found almost all actual pass students.
# So overall, it performs very well.


# ⚙️ 2️⃣ Technical Round – Deeper but Simple Explanation

# Precision = TP / (TP + FP) → Of all students predicted as pass, how many really passed.

# Recall = TP / (TP + FN) → Of all students who actually passed, how many were found by the model.

# F1-score = 2 × (Precision × Recall) / (Precision + Recall) → Balances both precision and recall, 
# especially when data is imbalanced (l
# ✅ Sample 20-Second Spoken Answer for Interview:

# In my project, I used precision, recall, and F1-score to evaluate 
# how well my model predicts student performance.

# Precision tells me how correct the model is when it predicts a student will pass.
# Recall tells me how many actual pass students it identified correctly.
# F1-score gives a balanced measure of both.

# My model’s precision and recall were both 0.98, which shows 
# it’s very reliable in predicting student success.