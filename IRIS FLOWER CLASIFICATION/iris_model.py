# iris_model.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk

# Step 1 – Load dataset directly from sklearn
iris = load_iris()
X = iris.data
y = iris.target_names[iris.target]

print(" Dataset Loaded Successfully!")
print(pd.DataFrame(X, columns=iris.feature_names).head())

# Step 2 – Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n Training and Testing Split Done!")

# Step 3 – Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("\n Model Training Completed!")

# Step 4 – Evaluate
y_pred = model.predict(X_test)
print("\n Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5 – Visualization
plt.figure(figsize=(6,4))
sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=y_pred, palette='Set2')
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Predicted Iris Species Distribution")
plt.show()

# Step 6 – Save model
pk.dump(model, open("iris_model.pkl", "wb"))
print("\n Model saved as iris_model.pkl")