import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load dataset
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


# Basic Info
print("\n First 10 records of dataset")
print(data.head(10))
print("\n Last 10 records of dataset")
print(data.tail(10))
print("\n Statistic summary ")
print(data.describe())
print("\n summary information")
print(data.info())
print("\n Number of rows and columns(shape)")
print(data.shape)

#check for missing values
print("\nCheck for missing values")
print(data.isnull().sum())

#outlier check
data1=data.drop(columns=["EmployeeNumber", "EmployeeCount", "StandardHours",
                         "Over18","Education","EnvironmentSatisfaction","JobInvolvement",
                         "JobLevel","JobSatisfaction","PerformanceRating",
                         "StockOptionLevel","WorkLifeBalance"])
numeric_cols = list(data1.select_dtypes(include=['int64', 'float64']).columns)
n_cols = 5
n_rows = math.ceil(len(numeric_cols) / n_cols)
plt.figure(figsize=(18, n_rows * 2))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(y=data[col], color="lightseagreen", fliersize=3)
    plt.title(col)
    
plt.tight_layout()
plt.show()


# EDA 

# 1. Attrition Count
plt.figure(figsize=(6,4))
sns.countplot(x="Attrition", hue="Attrition",data=data, palette="Set1")
plt.title("Attrition Count")
plt.tight_layout()
plt.show()

# 2. Attrition by Department
plt.figure(figsize=(8,5))
sns.countplot(x="Department", hue="Attrition", data=data, palette="magma")
plt.title("Attrition by Department")
plt.tight_layout()
plt.show()

# 3. Attrition by Job Role
plt.figure(figsize=(12,6))
sns.countplot(x="JobRole", hue="Attrition", data=data, palette="cividis")
plt.xticks(rotation=45)
plt.title("Attrition by Job Role")
plt.tight_layout()
plt.show()

# 4. Monthly Income vs Attrition
plt.figure(figsize=(8,5))
sns.boxplot(x="Attrition", y="MonthlyIncome", hue="Attrition",data=data, palette="Dark2")
plt.title("Monthly Income vs Attrition")
plt.tight_layout()
plt.show()

# 5. Age distribution by Attrition
plt.figure(figsize=(8,5))
sns.kdeplot(data=data, x="Age", hue="Attrition",palette=["purple","#FF69B4"], fill=True)
plt.title("Age Distribution by Attrition")
plt.tight_layout()
plt.show()

# 6. Years at Company vs Attrition
plt.figure(figsize=(8,5))
sns.histplot(data=data, x="YearsAtCompany", hue="Attrition", multiple="stack", bins=20,
             palette=["#9ED10F","#3BA500"],alpha=0.7)
plt.title("Years at Company vs Attrition")
plt.tight_layout()
plt.show()

# 7. Correlation Heatmap
numeric_df = data.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), cmap='Greens', annot=False)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.show()

# Data Preprocessing

# Encode target
data['Attrition'] = data['Attrition'].map({'Yes':1, 'No':0})

# Encode categorical variables
cat_cols = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

# Split data
X = data.drop("Attrition", axis=1)
y = data["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

# Model Training & Evaluation

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_proba):.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Feature Importance

# Random Forest Feature Importance
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
importances = pd.Series(rf_model.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=importances.values[:15], y=importances.index[:15],
            hue=importances.index[:15],legend=False, palette="viridis")
plt.title("Top 15 Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

print("\nTop Features from Random Forest:")
print(importances.head(15))
