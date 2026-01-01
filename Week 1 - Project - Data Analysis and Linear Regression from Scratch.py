# -----------------------------------------------------------
# LINEAR REGRESSION FROM SCRATCH + SCIKIT-LEARN COMPARISON
# -----------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------
# 1. Create Simple Dataset (House Prices)
# -----------------------------------------------------------

data = {
    "Area": [800, 900, 1000, 1100, 1200, 1500, 1800, 2000, 2200, 2500],
    "Price": [35, 40, 45, 48, 52, 60, 75, 85, 95, 110]
}
df = pd.DataFrame(data)
print("\nDataset:\n", df)

# -----------------------------------------------------------
# 2. Exploratory Data Analysis
# -----------------------------------------------------------

plt.scatter(df["Area"], df["Price"])
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakhs)")
plt.title("House Price vs Area")
plt.show()

# -----------------------------------------------------------
# 3. Prepare Data
# -----------------------------------------------------------

X = df["Area"].values.reshape(-1, 1)
y = df["Price"].values.reshape(-1, 1)

# -----------------------------------------------------------
# 4. Linear Regression from Scratch (Batch Gradient Descent)
# -----------------------------------------------------------

def gradient_descent(X, y, learning_rate=0.0000001, epochs=20000):
    m = 0
    c = 0
    n = len(X)

    for i in range(epochs):
        y_pred = m * X + c
        error = y_pred - y

        # Gradients
        dm = (2/n) * np.sum(error * X)
        dc = (2/n) * np.sum(error)

        # Update parameters
        m -= learning_rate * dm
        c -= learning_rate * dc

        if i % 2000 == 0:
            cost = np.mean(error ** 2)
            print(f"Epoch {i} | Cost: {cost:.4f} | m: {m:.4f} | c: {c:.4f}")

    return m, c

# Train Model
m, c = gradient_descent(X, y)

print("\n--------- Custom Linear Regression Results ---------")
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Custom Predictions
y_pred_custom = m * X + c

# -----------------------------------------------------------
# 5. Visualize Custom Regression Line
# -----------------------------------------------------------

plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred_custom, label="Custom Regression Line", linewidth=3)
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakhs)")
plt.title("Linear Regression Using Gradient Descent (Scratch)")
plt.legend()
plt.show()

# -----------------------------------------------------------
# 6. Scikit-Learn Linear Regression
# -----------------------------------------------------------

lr = LinearRegression()
lr.fit(X, y)
y_pred_sklearn = lr.predict(X)

print("\n--------- Scikit-Learn Results ---------")
print("Slope:", lr.coef_[0][0])
print("Intercept:", lr.intercept_[0])

# -----------------------------------------------------------
# 7. Compare Errors
# -----------------------------------------------------------

mse_custom = mean_squared_error(y, y_pred_custom)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)

print("\n--------- Model Comparison ---------")
print("MSE - Custom Model :", mse_custom)
print("MSE - Scikit-Learn:", mse_sklearn)
