import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv(r"C:\Users\Asus\Downloads\data.csv")   

# Remove non-numeric columns
data = data.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1)

# Drop rows with missing values (important)
data = data.dropna()

# Target column
y = data['price']
X = data.drop('price', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Example prediction
print("\nActual Price:", y_test.iloc[0])
print("Predicted Price:", model.predict([X_test.iloc[0]])[0])
