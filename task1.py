import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load CSV File
file_path = r"C:\Users\Asus\Downloads\IRIS.csv"
df = pd.read_csv(file_path)

# Features and Target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Encode species labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=300)
model.fit(X_train_scaled, y_train)

# Model Accuracy
pred = model.predict(X_test_scaled)
print("Model Accuracy:", accuracy_score(y_test, pred))

# Predict from user input
print("\nEnter flower measurements:")
sl = float(input("Sepal length: "))
sw = float(input("Sepal width : "))
pl = float(input("Petal length: "))
pw = float(input("Petal width : "))

# Convert user input to dataframe to avoid warnings
sample_df = pd.DataFrame([[sl, sw, pl, pw]], 
                         columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Scale properly
sample_scaled = scaler.transform(sample_df)

# Predict
prediction = model.predict(sample_scaled)
species_name = le.inverse_transform(prediction)[0]

print("\nPredicted Species:", species_name)
