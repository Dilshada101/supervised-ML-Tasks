import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

df = pd.read_csv("dataset_.csv", skiprows=41, header=None, names=columns)
df = df[df["CRIM"] != "@DATA"]
df.reset_index(drop=True, inplace=True)
df = df.apply(pd.to_numeric)


print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
# Step 4: Train Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test)

# Step 5: Evaluate Linear Regression
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_mae = mean_absolute_error(y_test, y_pred_linear)

print("\n========== Linear Regression ==========")
print("MSE :", linear_mse)
print("MAE :", linear_mae)


# Step 6: Residual Plot
residuals = y_test - y_pred_linear

plt.figure(figsize=(8,5))
plt.scatter(y_pred_linear, residuals)
plt.axhline(y=0, color='red')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot - Linear Regression")
plt.show()


# Step 7: Train Ridge Regression
ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)

print("\n========== Ridge Regression ==========")
print("MSE :", ridge_mse)
print("MAE :", ridge_mae)


# Step 8: Train Lasso Regression
lasso_model = Lasso(alpha=0.1)

lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)

lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)

print("\n========== Lasso Regression ==========")
print("MSE :", lasso_mse)
print("MAE :", lasso_mae)


# Step 9: Compare Metrics
metrics_df = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "MSE": [linear_mse, ridge_mse, lasso_mse],
    "MAE": [linear_mae, ridge_mae, lasso_mae]
})

print("\n========== Model Comparison ==========")
print(metrics_df)


# Step 10: Compare Coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Linear": linear_model.coef_,
    "Ridge": ridge_model.coef_,
    "Lasso": lasso_model.coef_
})

print("\n========== Coefficient Comparison ==========")
print(coef_df)

# Step 11: Plot Coefficients
coef_df.set_index("Feature").plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Coefficient Comparison")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.show()


