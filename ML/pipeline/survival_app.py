import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Titanic Survival Analysis", layout="wide")

st.title(" Titanic Death Factor Analysis")
st.markdown("Analyze which factors most contributed to deaths in the Titanic disaster")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Emphasis\FirstProject\ML\pipeline\tested.csv")

test = load_data()

# Sidebar for controls
st.sidebar.header("Analysis Controls")
show_raw_data = st.sidebar.checkbox("Show Raw Data")
show_age_distribution = st.sidebar.checkbox("Show Age Distribution")
show_survival_charts = st.sidebar.checkbox("Show Survival Charts")

# Main content
if show_raw_data:
    st.subheader(" Raw Dataset")
    st.dataframe(test.head(10))
    st.write(f"Dataset shape: {test.shape}")

# Age categorization function
def age_category(x):
    if x < 16:
        return 'child'
    elif x >= 16 and x < 40:
        return 'middle'
    else:
        return 'old'

# Process age data
age_data = test['Age'].fillna(0).values
age_df = pd.DataFrame(age_data)
age_df.loc[age_df[0] < 16, 'category'] = 'child'
age_df.loc[(age_df[0] >= 16) & (age_df[0] < 40), 'category'] = 'middle'
age_df.loc[age_df[0] >= 40, 'category'] = 'old'
age_df[0] = age_df[0].apply(age_category)

# Add age category to main dataset
test['Age_Category'] = age_df[0]

# Age distribution
if show_age_distribution:
    st.subheader("👥 Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    age_counts = test['Age_Category'].value_counts()
    ax.pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Passenger Age Distribution")
    st.pyplot(fig)

# Survival analysis
st.subheader("� Death Factor Analysis")

# Calculate death rates (1 - survival rate)
age_death = 1 - test.groupby('Age_Category')['Survived'].mean()
sex_death = 1 - test.groupby('Sex')['Survived'].mean()
class_death = 1 - test.groupby('Pclass')['Survived'].mean()

# Fare categories
test['Fare_Category'] = pd.cut(test['Fare'].fillna(0), bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
fare_death = 1 - test.groupby('Fare_Category')['Survived'].mean()

# Display death rates in columns
col1, col2 = st.columns(2)

with col1:
    st.write("**Death Rate by Age Category**")
    for age_cat, rate in age_death.items():
        st.write(f"{age_cat}: {rate:.2%}")
    
    st.write("**Death Rate by Sex**")
    for sex, rate in sex_death.items():
        st.write(f"{sex}: {rate:.2%}")

with col2:
    st.write("**Death Rate by Passenger Class**")
    for pclass, rate in class_death.items():
        st.write(f"Class {pclass}: {rate:.2%}")
    
    st.write("**Death Rate by Fare Category**")
    for fare_cat, rate in fare_death.items():
        st.write(f"{fare_cat}: {rate:.2%}")

# Death charts
if show_survival_charts:
    st.subheader("📈 Death Rate Visualization")
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Age death
    age_death.plot(kind='bar', ax=ax1, color='red', alpha=0.7)
    ax1.set_title('Death Rate by Age')
    ax1.set_ylabel('Death Rate')
    ax1.tick_params(axis='x', rotation=45)
    
    # Sex death
    sex_death.plot(kind='bar', ax=ax2, color='darkred', alpha=0.7)
    ax2.set_title('Death Rate by Sex')
    ax2.set_ylabel('Death Rate')
    ax2.tick_params(axis='x', rotation=45)
    
    # Class death
    class_death.plot(kind='bar', ax=ax3, color='crimson', alpha=0.7)
    ax3.set_title('Death Rate by Class')
    ax3.set_ylabel('Death Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    # Fare death
    fare_death.plot(kind='bar', ax=ax4, color='firebrick', alpha=0.7)
    ax4.set_title('Death Rate by Fare')
    ax4.set_ylabel('Death Rate')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

# Factor influence analysis
st.subheader(" Death Factor Ranking")

# ML Model for Coefficient Analysis
st.subheader(" Logistic Regression Coefficient Analysis")

# Prepare data for ML model
@st.cache_data
def prepare_ml_data():
    # Create a copy for ML processing
    ml_data = test.copy()
    
    # Handle missing values
    ml_data['Age'] = ml_data['Age'].fillna(ml_data['Age'].median())
    ml_data['Fare'] = ml_data['Fare'].fillna(ml_data['Fare'].median())
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    ml_data['Sex_encoded'] = le_sex.fit_transform(ml_data['Sex'])
    
    # Create age categories
    ml_data['Age_cat'] = ml_data['Age'].apply(age_category)
    le_age = LabelEncoder()
    ml_data['Age_encoded'] = le_age.fit_transform(ml_data['Age_cat'])
    
    # Select features for ML
    features = ['Age_encoded', 'Sex_encoded', 'Pclass', 'Fare']
    X = ml_data[features]
    y = ml_data['Survived']  # 0 = died, 1 = survived
    
    return X, y, features, le_sex, le_age

X, y, feature_names, le_sex, le_age = prepare_ml_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
@st.cache_data
def train_model():
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    return lr_model

lr_model = train_model()

# Calculate coefficients
st.write("**📈 Logistic Regression Death Factor Coefficients**")
lr_coef = np.abs(lr_model.coef_[0])  # Absolute values for importance
lr_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr_coef
}).sort_values('Coefficient', ascending=False)
st.dataframe(lr_df, use_container_width=True)

# Visualize Logistic Regression coefficients
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(lr_df['Feature'], lr_df['Coefficient'], color='blue', alpha=0.7)
ax.set_xlabel('Absolute Coefficient Value')
ax.set_title('Logistic Regression - Death Factor Coefficients')
st.pyplot(fig)

# Model accuracy
st.write("**🎯 Model Performance**")
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
st.write(f"Logistic Regression Accuracy: {lr_acc:.3f}")

# Condition Number Analysis
st.subheader("🔢 Multicollinearity Analysis")

# Calculate condition number
from numpy.linalg import cond
condition_number = cond(X_train)
st.write(f"**Condition Number:** {condition_number:.2f}")

# Interpret condition number
if condition_number < 10:
    st.write("✅ **Good**: No multicollinearity issues")
elif condition_number < 30:
    st.write("⚠️ **Moderate**: Some multicollinearity present")
elif condition_number < 100:
    st.write("🟡 **Concerning**: Significant multicollinearity")
else:
    st.write("🔴 **Severe**: High multicollinearity - consider feature selection")

# Feature correlation matrix
st.write("**📊 Feature Correlation Matrix:**")
correlation_matrix = X_train.corr()
st.dataframe(correlation_matrix.round(3), use_container_width=True)

# Visualize correlation
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Feature Correlation Heatmap')
st.pyplot(fig)

# Combined analysis
st.subheader("🔍 Coefficient vs Statistical Analysis Comparison")

# Calculate statistical death factors (original method)
death_factors = {
    'Age': age_death.max() - age_death.min(),
    'Sex': sex_death.max() - sex_death.min(),
    'Class': class_death.max() - class_death.min(),
    'Fare': fare_death.max() - fare_death.min()
}

# Create comparison
combined_df = pd.DataFrame({
    'Feature': feature_names,
    'Statistical_Impact': [death_factors['Age'], death_factors['Sex'], death_factors['Class'], death_factors['Fare']],
    'LR_Coefficient': lr_coef
})

# Normalize both scores to 0-1 scale for comparison
combined_df['Statistical_Norm'] = combined_df['Statistical_Impact'] / combined_df['Statistical_Impact'].max()
combined_df['LR_Norm'] = combined_df['LR_Coefficient'] / combined_df['LR_Coefficient'].max()
combined_df['Average_Score'] = (combined_df['Statistical_Norm'] + combined_df['LR_Norm']) / 2

# Sort by average score
combined_df = combined_df.sort_values('Average_Score', ascending=False)

st.write("**🏆 Final Death Factor Ranking (Combined Methods):**")
st.dataframe(combined_df[['Feature', 'Average_Score']].round(3), use_container_width=True)

# Visualize comparison
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(combined_df['Feature']))
width = 0.35

ax.bar(x_pos - width/2, combined_df['Statistical_Norm'], width, label='Statistical Method', alpha=0.7)
ax.bar(x_pos + width/2, combined_df['LR_Norm'], width, label='Logistic Regression', alpha=0.7)

ax.set_xlabel('Features')
ax.set_ylabel('Normalized Importance Score')
ax.set_title('Death Factor Importance - Statistical vs Logistic Regression')
ax.set_xticks(x_pos)
ax.set_xticklabels(combined_df['Feature'])
ax.legend()
ax.tick_params(axis='x', rotation=45)

st.pyplot(fig)

# Use Logistic Regression coefficients as the primary method
sorted_factors = sorted(zip(feature_names, lr_coef), key=lambda x: x[1], reverse=True)

# Display results
col1, col2, col3 = st.columns(3)

with col1:
    st.error(f"**Highest Death Factor:** {sorted_factors[0][0]}")
    st.write(f"Coefficient: {sorted_factors[0][1]:.3f}")

with col2:
    if len(sorted_factors) > 2:
        st.warning(f"**Middle Factor:** {sorted_factors[1][0]}")
        st.write(f"Coefficient: {sorted_factors[1][1]:.3f}")

with col3:
    st.success(f"**Lowest Death Factor:** {sorted_factors[-1][0]}")
    st.write(f"Coefficient: {sorted_factors[-1][1]:.3f}")

# Complete ranking
st.write("**Complete Death Factor Ranking (Logistic Regression Coefficients):**")
ranking_df = pd.DataFrame(sorted_factors, columns=['Factor', 'LR Coefficient'])
ranking_df['LR Coefficient'] = ranking_df['LR Coefficient'].round(3)
st.dataframe(ranking_df, use_container_width=True)

# Key insights
st.subheader("💡 Key Insights")
most_deadly = sorted_factors[0][0]
least_deadly = sorted_factors[-1][0]

if most_deadly == 'Sex_encoded':
    st.write(" **Gender was the deadliest factor** - Men had significantly higher death rates than women")
elif most_deadly == 'Pclass':
    st.write(" **Class was the deadliest factor** - Third-class passengers had much higher death rates")
elif most_deadly == 'Age_encoded':
    st.write(" **Age was the deadliest factor** - Adults had higher death rates than children")
else:
    st.write(" **Fare was the deadliest factor** - Lower fare passengers had higher death rates")

st.write(f" **{least_deadly} had the minimal impact** on death rates")

st.markdown("---")
st.markdown("*This analysis shows how different factors contributed to deaths in the Titanic disaster.*")
