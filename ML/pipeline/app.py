import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import statsmodels.api as sm
from numpy.linalg import cond
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load Titanic dataset
print("Loading Titanic dataset...")
test = pd.read_csv(r"C:\Emphasis\FirstProject\ML\pipeline\tested.csv")
print(f"Dataset shape: {test.shape}")

# Data preprocessing
print("\n=== Data Preprocessing ===")

# Handle missing values
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Age categorization function
def age_category(x):
    if x < 16:
        return 'child'
    elif x >= 16 and x < 40:
        return 'middle'
    else:
        return 'old'

test['Age_Category'] = test['Age'].apply(age_category)

# OneHotEncoding for categorical variables
print("Applying OneHotEncoding with drop='first' to avoid multicollinearity...")

# OneHotEncode Sex (drop='first' to avoid multicollinearity)
ohe_sex = OneHotEncoder(sparse_output=False, drop='first')
sex_encoded = ohe_sex.fit_transform(test[['Sex']])
sex_df = pd.DataFrame(sex_encoded, columns=['Sex_male'])

# OneHotEncode Age Category (drop='first' to avoid multicollinearity)
ohe_age = OneHotEncoder(sparse_output=False, drop='first')
age_encoded = ohe_age.fit_transform(test[['Age_Category']])
age_df = pd.DataFrame(age_encoded, columns=['Age_middle', 'Age_old'])

# OneHotEncode Pclass (drop='first' to avoid multicollinearity)
ohe_pclass = OneHotEncoder(sparse_output=False, drop='first')
pclass_encoded = ohe_pclass.fit_transform(test[['Pclass']])
pclass_df = pd.DataFrame(pclass_encoded, columns=['Class_2', 'Class_3'])

# Combine all features (avoiding perfect multicollinearity)
features_df = pd.concat([
    sex_df, age_df, pclass_df, 
    test[['Fare']].reset_index(drop=True)
], axis=1)

print(f"Features shape after OneHotEncoding: {features_df.shape}")
print("Feature columns:", list(features_df.columns))

# Prepare data for GLM
X = features_df
y = test['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Train OLS model
print("\n=== Training OLS Model ===")

# Add constant for intercept
X_train_const = sm.add_constant(X_train)

# Fit OLS model
ols_model = sm.OLS(y_train, X_train_const)
ols_results = ols_model.fit()

print("OLS model trained successfully!")
print(f"R-squared: {ols_results.rsquared:.3f}")
print(f"Adjusted R-squared: {ols_results.rsquared_adj:.3f}")
print(f"F-statistic: {ols_results.fvalue:.3f}")
print(f"AIC: {ols_results.aic:.3f}")
print(f"BIC: {ols_results.bic:.3f}")

# Make predictions
X_test_const = sm.add_constant(X_test)
y_pred = ols_results.predict(X_test_const)
y_pred_class = (y_pred > 0.5).astype(int)

# Model performance
accuracy = accuracy_score(y_test, y_pred_class)
print(f"\nModel Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

# Display full model summary
print("\n=== OLS Model Summary ===")
print(ols_results.summary())

# Coefficient analysis
print("\n=== Coefficient Analysis ===")

# Get coefficients and statistics from OLS results
coefficients = ols_results.params[1:]  # Exclude intercept
p_values = ols_results.pvalues[1:]  # Exclude intercept
conf_int = ols_results.conf_int().iloc[1:]  # Exclude intercept

# Create results dataframe
results_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'P-Value': p_values,
    'CI_Lower': conf_int.iloc[:, 0],
    'CI_Upper': conf_int.iloc[:, 1],
    'Significant': p_values < 0.05
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nGLM Coefficients:")
print(results_df.round(4))

# Significant features
significant_features = results_df[results_df['Significant']]
print(f"\nSignificant Features (p < 0.05): {len(significant_features)}")
print(significant_features[['Feature', 'Coefficient', 'P-Value']].round(4))

# Multicollinearity analysis
print("\n=== Multicollinearity Analysis ===")

# Calculate condition number
condition_number = cond(X_train)
print(f"Condition Number: {condition_number:.2f}")

if condition_number < 10:
    print("✅ Good: No multicollinearity issues")
elif condition_number < 30:
    print("⚠️ Moderate: Some multicollinearity present")
elif condition_number < 100:
    print("🟡 Concerning: Significant multicollinearity")
else:
    print("🔴 Severe: High multicollinearity - consider feature selection")

# VIF Analysis
print("\nVIF Analysis:")
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_const.values, i) 
                    for i in range(X_train_const.shape[1])]

# Remove constant row for display
vif_data = vif_data[vif_data["Feature"] != "const"]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.round(3))

# High VIF features
high_vif_features = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
if high_vif_features:
    print(f"\n🚨 Features with high VIF (>5): {high_vif_features}")
else:
    print("\n✅ All features have acceptable VIF values (<5)")

# Statistical death rate analysis
print("\n=== Statistical Death Rate Analysis ===")

# Calculate death rates by original categories
death_rates = {}

# Death rate by sex
sex_death = 1 - test.groupby('Sex')['Survived'].mean()
death_rates['Sex'] = sex_death.max() - sex_death.min()
print(f"Death rate by Sex: {sex_death.to_dict()}")

# Death rate by age category
age_death = 1 - test.groupby('Age_Category')['Survived'].mean()
death_rates['Age'] = age_death.max() - age_death.min()
print(f"Death rate by Age Category: {age_death.to_dict()}")

# Death rate by class
class_death = 1 - test.groupby('Pclass')['Survived'].mean()
death_rates['Class'] = class_death.max() - class_death.min()
print(f"Death rate by Class: {class_death.to_dict()}")

# Death rate by fare (create categories)
test['Fare_Category'] = pd.cut(test['Fare'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
fare_death = 1 - test.groupby('Fare_Category')['Survived'].mean()
death_rates['Fare'] = fare_death.max() - fare_death.min()
print(f"Death rate by Fare Category: {fare_death.to_dict()}")

print(f"\nDeath Rate Differences: {death_rates}")

# Feature importance ranking
print("\n=== Feature Importance Ranking ===")

# Map OneHotEncoded features back to original categories
feature_importance = {}

for feature in X.columns:
    if feature == 'Sex_male':
        category = 'Sex'
    elif feature.startswith('Age_'):
        category = 'Age'
    elif feature.startswith('Class_'):
        category = 'Class'
    elif feature == 'Fare':
        category = 'Fare'
    else:
        continue
    
    # Get coefficient for this feature
    coef = results_df[results_df['Feature'] == feature]['Coefficient'].values[0]
    feature_importance[category] = abs(coef)

print("Feature Importance by Category:")
for category, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{category}: {importance:.4f}")

# Final death factor ranking
print("\n=== Final Death Factor Ranking ===")
most_important = max(feature_importance, key=feature_importance.get)
least_important = min(feature_importance, key=feature_importance.get)

print(f"Most deadly factor: {most_important}")
print(f"Least deadly factor: {least_important}")

print(f"\nComplete ranking (most to least deadly):")
for category, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{category}: {importance:.4f}")

# Visualization
print("\n=== Creating Visualizations ===")

# Create coefficient plot
plt.figure(figsize=(12, 8))
significant_mask = results_df['Significant']
colors = ['red' if sig else 'lightblue' for sig in significant_mask]

plt.barh(results_df['Feature'], results_df['Coefficient'], color=colors, alpha=0.7)
plt.xlabel('Coefficient Value')
plt.title('GLM Coefficients (Red = Significant, Blue = Not Significant)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('glm_coefficients.png', dpi=300, bbox_inches='tight')
plt.clf()

# Create death rate comparison chart
plt.figure(figsize=(10, 6))
categories = list(death_rates.keys())
rates = list(death_rates.values())

plt.bar(categories, rates, color='red', alpha=0.7)
plt.xlabel('Categories')
plt.ylabel('Death Rate Difference')
plt.title('Death Rate Differences by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('death_rates.png', dpi=300, bbox_inches='tight')
plt.clf()

# Create correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = X_train.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.clf()

print("Visualizations saved: glm_coefficients.png, death_rates.png, correlation_heatmap.png")

# Summary
print("\n=== SUMMARY ===")
print(f"Model: OLS (Ordinary Least Squares)")
print(f"Accuracy: {accuracy:.3f}")
print(f"R-squared: {ols_results.rsquared:.3f}")
print(f"Adjusted R-squared: {ols_results.rsquared_adj:.3f}")
print(f"Condition Number: {condition_number:.2f}")
print(f"Significant Features: {len(significant_features)}/{len(results_df)}")
print(f"Most Deadly Factor: {most_important}")
print(f"Least Deadly Factor: {least_important}")

if condition_number < 30 and len(high_vif_features) == 0:
    print(" Model is optimal: Good statistical properties")
elif condition_number < 30:
    print(" Consider feature selection: Some issues detected")
else:
    print(" Feature selection needed: High multicollinearity")

print("\nAnalysis complete!")
print("\n=== FINAL OLS RESULTS ===")
print(ols_results.summary())
