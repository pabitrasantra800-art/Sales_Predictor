import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_excel(r"C:\Users\pabit\Downloads\Sales_Forecasting_Dataset (1).xlsx")

X = data[["Month","Advertising","Price","Competitor_Price"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=200,random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Predicted Sales:",y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("MAE:",mae)
print("R2 Score:",r2)

future = pd.DataFrame({
    "Month":[13,14,15],
    "Advertising":[5200,5400,5600],
    "Price":[14,13,13],
    "Competitor_Price":[16,16,15]
})

future_sales = model.predict(future)

print("Future Sales Prediction:")
print(future_sales)

# Plot with different colours
plt.figure(figsize=(7,5))

plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red",
         label="Perfect Prediction Line")

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Random Forest Sales Forecasting")
plt.legend()

plt.show()
