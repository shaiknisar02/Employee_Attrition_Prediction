import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import pickle

# Load dataset
df = pd.read_csv("HR_Employee_Attrition.csv")

# Drop less informative columns
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Exploratory Data Analysis
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Encode categorical features
le_dict = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    le_dict[column] = le

# Feature Scaling (optional but helpful for Logistic Regression)
scaler = StandardScaler()
X = df.drop('Attrition', axis=1)
X_scaled = scaler.fit_transform(X)

y = df['Attrition']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Cross-validation score
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print("\nCross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Evaluation
y_pred = model.predict(X_test)
print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC
y_prob = model.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_score)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Feature Importance (coefficients)
feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(14, 6), title='Feature Importance (Logistic Coefficients)')
plt.tight_layout()
plt.show()

# Save the model, feature columns, encoders, and scaler
with open("attrition_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(le_dict, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Example Prediction
sample = X_test[0]
pred = model.predict([sample])
prob = model.predict_proba([sample])[0][1]
print(f"\nSample Prediction: {'Yes' if pred[0]==1 else 'No'} (Probability: {prob:.2f})")
