import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset/essays.csv")

# Inputs and outputs
X = data["essay"]
y = data["score"]

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words='english')

X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/essay_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model trained successfully!")