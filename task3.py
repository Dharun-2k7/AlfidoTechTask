import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset (change path)
data = pd.read_csv(r"C:\Users\Asus\Downloads\titanic.csv")

print("Dataset Loaded Successfully!")
print(data.head())

# 2. Drop text-heavy or mostly empty columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']

for col in columns_to_drop:
    if col in data.columns:
        data = data.drop(col, axis=1)

# 3. Fill missing values
# Age → median
if 'Age' in data.columns:
    data['Age'] = data['Age'].fillna(data['Age'].median())

# Fare → median
if 'Fare' in data.columns:
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Embarked → mode
if 'Embarked' in data.columns:
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# 4. Encode categorical variables
label_enc = LabelEncoder()

for col in ['Sex', 'Embarked']:
    if col in data.columns:
        data[col] = label_enc.fit_transform(data[col])

# 5. Separate features and target
y = data['Survived']
X = data.drop('Survived', axis=1)

# 6. Final check for NaN → remove if found
X = X.fillna(0)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 9. Predictions
y_pred = model.predict(X_test)

# 10. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
