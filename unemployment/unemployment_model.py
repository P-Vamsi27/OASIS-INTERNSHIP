# unemployment_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle as pk

# Step 1: Load dataset
data = pd.read_csv(r"V:\OASIS INTERNSHIP\unemployment\Unemployment in India.csv")
data.columns = data.columns.str.strip()  # clean column names
print(" Dataset Loaded Successfully!")
print(data.head())

# Step 2: Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')  # safer parsing
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data.dropna(inplace=True)

# Drop the raw Date column (not usable for ML)
data.drop('Date', axis=1, inplace=True)

# Encode categorical columns
data = pd.get_dummies(data, drop_first=True)
print("\n Data after encoding:\n", data.head())

# Step 3: Define features and target
X = data.drop('Estimated Unemployment Rate (%)', axis=1)
y = data['Estimated Unemployment Rate (%)']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n Training and Testing Split Done!")

# Step 5: Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("\n Model Training Completed!")

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("\n Model Evaluation:")
print("R2 Score:", round(r2, 3))
print("Mean Absolute Error:", round(mae, 3))

# Step 8: Visualization
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # reference line
plt.xlabel("Actual Unemployment Rate")
plt.ylabel("Predicted Unemployment Rate")
plt.title("Actual vs Predicted Unemployment Rate")
plt.show()

# Step 9: Save model
pk.dump(model, open("unemployment_model.pkl", "wb"))
print("\n Model saved as unemployment_model.pkl")
