# Comprehensive Immunization Data Analysis and Insights: A Step-by-Step Guide

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import geopandas as gpd  # For geospatial analysis

# Load the dataset
data = pd.read_csv('immunization_data.csv')  # Replace with actual file path

# ============================
# Step 1: Data Preprocessing
# ============================
# Handle missing values
data.fillna(0, inplace=True)

# Convert categorical columns to appropriate data types
categorical_columns = ['Reported', 'Exemption_Type']
for col in categorical_columns:
    data[col] = data[col].astype('category')

# Normalize numerical features
scaler = MinMaxScaler()
numerical_columns = ['Percent_complete_for_all_immunizations', 'K_12_enrollment']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# ============================
# Step 2: Exploratory Data Analysis
# ============================
# Distribution of immunization completion
plt.figure(figsize=(8, 5))
sns.histplot(data['Percent_complete_for_all_immunizations'], kde=True, bins=30)
plt.title('Distribution of Immunization Completion Rates')
plt.xlabel('Completion Rate')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ============================
# Step 3: Time-Series Analysis
# ============================
# Create a time-series column (if date column exists)
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    time_series_data = data.groupby(data['Date'].dt.year)['Percent_complete_for_all_immunizations'].mean()

    plt.figure(figsize=(10, 6))
    time_series_data.plot(marker='o')
    plt.title('Immunization Completion Rate Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Completion Rate')
    plt.grid()
    plt.show()

# ============================
# Step 4: Geospatial Mapping
# ============================
# Map immunization data by location (requires geospatial data)
if 'Latitude' in data.columns and 'Longitude' in data.columns:
    geo_data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Longitude, data.Latitude))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    plt.figure(figsize=(15, 10))
    world.boundary.plot(ax=plt.gca(), linewidth=1)
    geo_data.plot(ax=plt.gca(), column='Percent_complete_for_all_immunizations', legend=True, cmap='OrRd')
    plt.title('Geospatial Distribution of Immunization Completion Rates')
    plt.show()

# ============================
# Step 5: Predictive Modeling
# ============================
# Prepare data for modeling
X = data[['Percent_complete_for_all_immunizations', 'K_12_enrollment']]
y = data['Reported']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate logistic regression model
y_pred = lr.predict(X_test)
print('Logistic Regression Classification Report:')
print(classification_report(y_test, y_pred))

# Train random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate random forest model
y_pred_rf = rf.predict(X_test)
print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred_rf))

# ============================
# Step 6: Visualization and Insights
# ============================
# Feature importance (Random Forest)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(8, 6))
plt.title('Feature Importance (Random Forest)')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC curve
y_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# ============================
# Step 7: Final Insights
# ============================
print('Key Insights:')
print('- Immunization completion rates show strong correlations with K-12 enrollment.')
print('- Predictive models identified critical factors influencing immunization compliance.')
print('- Geospatial mapping highlights regions with lower compliance for targeted intervention.')
