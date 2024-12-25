# Comprehensive Immunization Data Analysis and Insights: A Step-by-Step Guide

# Immunization Data by School: Analysis and Insights

## Purpose
This project aims to identify patterns, trends, and anomalies in school immunization data to support public health initiatives. The analysis helps in understanding factors influencing immunization rates and exemptions, enabling data-driven decision-making to improve public health outcomes and compliance rates.

---

## Exceptional Achievements
1. **Data Consolidation and Standardization**:
   - Processed and standardized a complex dataset, including handling diverse data types and missing values, ensuring data integrity and usability.
   - Introduced robust feature scaling techniques (`StandardScaler` and `MinMaxScaler`) to normalize data, enhancing model reliability.

2. **Advanced Statistical Analysis**:
   - Conducted Chi-Square tests to establish statistically significant relationships between variables, revealing critical insights about immunization compliance.
   - Demonstrated exceptional strength in identifying relationships between immunization completion rates and key factors (Cramer’s V = 1.0 in one test).

3. **Feature Importance with Logistic Regression**:
   - Utilized logistic regression to highlight key predictors, offering actionable insights into factors impacting immunization compliance.

4. **Interactive Multivariate Analysis with Tableau**:
   - Developed visually compelling, interactive dashboards in Tableau to allow stakeholders to explore multivariate relationships dynamically.

5. **Actionable Recommendations**:
   - Provided insights that can directly inform policy and interventions, potentially increasing immunization compliance rates and reducing medical exemptions.

---

## Step-by-Step Process

### Step 1: Data Preprocessing
- **Loading Data**: Imported immunization data for cleaning and preparation.
- **Cleaning Steps**:
  - Removed duplicate rows to ensure data consistency.
  - Corrected data types for attributes.
  - Imputed missing values with contextually appropriate methods.
  - Standardized and normalized features using `StandardScaler` and `MinMaxScaler` for consistent scaling.

---

### Step 2: Exploratory Data Analysis (EDA)
- **Objective**: Uncover trends, patterns, and outliers.
- **Visualizations**:
  - Scatter plots, bar plots, and histograms to analyze distributions and relationships.

#### Example Visualization:
- **Top 10 Features by Logistic Regression Coefficients**:
  - A bar chart displaying the most significant predictors of immunization compliance based on logistic regression.

---

### Step 3: Modeling
- **Logistic Regression**:
  - Fitted a model to classify outcomes and identify influential predictors.
  - Used feature importance to prioritize variables impacting immunization rates.
- **Evaluation**:
  - Assessed accuracy and performance using confusion matrices and precision-recall metrics.

---

### Step 4: Inferential Statistics
Performed Chi-Square tests to investigate relationships between categorical variables.

#### Chi-Square Test Results:
1. **Reported vs. K-12 Enrollment**:
   - Pearson Chi-square = 2361.46
   - p-value = 0.0000
   - Cramer’s V = 0.9539 (strong association).

2. **Reported vs. Percent Complete for All Immunizations**:
   - Pearson Chi-square = 2595.0
   - p-value = 0.0000
   - Cramer’s V = 1.0 (very strong association).

3. **Reported vs. Number with Medical Exemptions**:
   - Pearson Chi-square = 300.85
   - p-value = 0.0000
   - Cramer’s V = 0.3405 (moderate association).

---

### Step 5: Advanced Visualization with Tableau
- Developed interactive Tableau dashboards to visualize multivariate relationships dynamically.
- Enabled stakeholders to explore key trends such as exemption hotspots and immunization compliance at a granular level.

---

### Step 6: Insights and Recommendations
- **Insights**:
  - High immunization completion rates are strongly associated with specific enrollment demographics.
  - Medical exemptions, though relatively rare, show significant variation across schools.
- **Recommendations**:
  - Target specific schools or districts with low compliance for tailored public health campaigns.
  - Review exemption policies in regions with high exemption rates.

---

## Exceptional Value Delivered
- This analysis provides actionable insights to public health officials, enabling focused interventions to improve immunization rates.
- The work emphasizes the importance of data-driven policy-making, offering a scalable framework for analyzing similar datasets in other regions or contexts.

---

## Tools and Libraries Used
- **Python**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- **Statistical Tests**: Chi-Square from the `researchpy` library.
- **Visualization Tools**: Tableau for advanced visualizations.

---

## Next Steps
- Share Tableau dashboards with stakeholders for actionable decision-making.
- Integrate findings into public health initiatives to improve compliance and reduce exemption rates.
- Expand the analysis to include temporal trends and geospatial mapping for deeper insights.
