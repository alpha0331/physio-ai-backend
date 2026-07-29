import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv('exercise_data.csv')
print(f"Loaded {len(df)} rows")
print(df['label'].value_counts())

# Features = the 4 angle columns, Target = label
X = df[['elbow_angle', 'shoulder_angle', 'hip_angle', 'knee_angle']]
y = df['label']

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining on {len(X_train)} rows, testing on {len(X_test)} rows")

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print("\nDetailed report:")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, 'exercise_classifier.pkl')
print("\nModel saved as exercise_classifier.pkl")