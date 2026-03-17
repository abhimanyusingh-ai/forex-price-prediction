import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Forex data
data = pd.read_csv("forex_data.csv")

# Create target: 1 if next day's price goes up, 0 if down
data["Target"] = (data["Price"].shift(-1) > data["Price"]).astype(int)

# Feature: daily returns
data["Returns"] = data["Price"].pct_change()

# Drop missing values
data = data.dropna()

# Define features and target
X = data[["Returns"]]
y = data["Target"]

# Train-test split (no shuffle to preserve time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
