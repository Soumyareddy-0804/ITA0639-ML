import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset
data = pd.DataFrame({
    'make': ['Toyota', 'Honda', 'BMW', 'Toyota', 'Honda'],
    'model': ['Corolla', 'Civic', 'X5', 'Camry', 'Accord'],
    'year': [2015, 2016, 2018, 2019, 2020],
    'mileage': [50000, 30000, 15000, 20000, 10000],
    'price': [15000, 16000, 35000, 25000, 27000]
})

# Data Preprocessing
data = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data.drop('price', axis=1)
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plotting the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Predict the price of a new car
new_car = pd.DataFrame([{
    'year': 2021,
    'mileage': 5000,
    'make_Honda': 1,
    'make_Toyota': 0,
    'model_Camry': 0,
    'model_Civic': 0,
    'model_Corolla': 0,
    'model_X5': 0
}])

# Ensure new_car has the same columns as X
new_car = new_car.reindex(columns=X.columns, fill_value=0)

predicted_price = model.predict(new_car)
print(f'Predicted Price: {predicted_price[0]}')
