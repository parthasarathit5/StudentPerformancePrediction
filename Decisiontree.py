# 🎯 Decision Tree Example - Supervised Learning (Classification)

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60],
    'Income': [25000, 40000, 80000, 110000, 95000, 120000, 90000, 130000],
    'BuysCar': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Features (Input)
X = df[['Age', 'Income']]
# Output (Label)
y = df['BuysCar']

# Create Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# Predict
prediction = model.predict([[30, 50000]])
print("Prediction for 30 years, income 50000:", prediction[0])

# Visualize tree
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=['Age', 'Income'], class_names=['No', 'Yes'], filled=True)
plt.show()
