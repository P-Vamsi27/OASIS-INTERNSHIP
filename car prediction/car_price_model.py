# car_price_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle as pk

# Step 1: Load dataset
data = pd.read_csv(R"V:\OASIS INTERNSHIP\car prediction\vamsi_car_data.csv")
print(" Dataset Loaded Successfully!")
print(data.head())

# Step 2: Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Step 3: Feature Engineering
data['Car_Age'] = 2026 - data['Year']
data.drop('Year', axis=1, inplace=True)

# Step 4: Encode categorical variables
data = pd.get_dummies(data, drop_first=True)
print("\n Data after encoding:\n", data.head())

# Step 5: Split features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n Training and Testing Split Done!")

# Step 7: Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("\n Model Training Completed!")

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("\n Model Evaluation:")
print("R2 Score:", round(r2, 3))
print("Mean Absolute Error:", round(mae, 3))

# Step 10: Visualization
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# Step 11: Save model
pk.dump(model, open("model.pkl", "wb"))
print("\n Model saved as model.pkl")
