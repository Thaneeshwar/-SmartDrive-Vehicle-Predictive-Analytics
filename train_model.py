import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load and prepare data
df = pd.read_csv("car data dekho.csv")

# Basic cleaning (if needed)
df = df.dropna(subset=['Selling_Price', 'Present_Price'])

# Features & target - most logical strong linear relationship
X = df[['Present_Price']]          # most important predictor
y = df['Selling_Price']

# 2. Create and train model
model = LinearRegression()
model.fit(X, y)

# 3. Predictions on training data
y_pred = model.predict(X)

# 4. Show model equation
print(f"Equation of line:")
print(f"Selling_Price = {model.coef_[0]:.3f} × Present_Price + {model.intercept_:.3f}")

# 5. Model performance metrics (much more informative than just R²)
print("\nModel Performance:")
print(f"R² score (how well model explains variance) : {r2_score(y, y_pred):.4f}")
print(f"Mean Absolute Error (average error in ₹ lakh) : {mean_absolute_error(y, y_pred):.3f}")
print(f"Root Mean Squared Error (typical error size) : {np.sqrt(mean_squared_error(y, y_pred)):.3f}")

# 6. Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.6, label="Actual cars")
plt.plot(X, y_pred, color="red", linewidth=2.5, label="Linear Regression fit")
plt.xlabel("Present Price (₹ lakh)")
plt.ylabel("Selling Price (₹ lakh)")
plt.title("Linear Regression: Present Price vs Selling Price\n(car data dekho dataset)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 7. Predict for new cars
new_prices = np.array([[5.0], [12.5], [25.0], [40.0]])  # example Present_Price values

predicted_selling = model.predict(new_prices)

print("\nPredictions for new cars:")
for present, sell in zip(new_prices.flatten(), predicted_selling):
    print(f"Present Price = ₹ {present:5.2f} lakh → Predicted Selling Price = ₹ {sell:6.2f} lakh")
