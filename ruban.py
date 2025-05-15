# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Step 1: Generate Synthetic Dataset
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'gender': np.random.choice([0, 1], size=n_samples),
    'SeniorCitizen': np.random.choice([0, 1], size=n_samples),
    'Partner': np.random.choice([0, 1], size=n_samples),
    'Dependents': np.random.choice([0, 1], size=n_samples),
    'tenure': np.random.randint(1, 72, size=n_samples),
    'PhoneService': np.random.choice([0, 1], size=n_samples),
    'InternetService': np.random.choice([0, 1, 2], size=n_samples),  # 0: No, 1: DSL, 2: Fiber
    'MonthlyCharges': np.random.uniform(20, 120, size=n_samples),
    'TotalCharges': lambda x: x['tenure'] * x['MonthlyCharges'],
    'Contract': np.random.choice([0, 1, 2], size=n_samples),  # 0: Month-to-month, 1: One year, 2: Two year
    'PaymentMethod': np.random.choice([0, 1, 2, 3], size=n_samples),
    'Churn': np.random.choice([0, 1], size=n_samples, p=[0.73, 0.27])  # Imbalanced churn
})
df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']

print("✅ Synthetic dataset created.\n", df.head())

# Step 2: Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Build and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predictions and Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 7: Feature Importance
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
