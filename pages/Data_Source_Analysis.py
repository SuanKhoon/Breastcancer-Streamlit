import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

# Create 'charts' folder if it doesn't exist
if not os.path.exists('charts'):
    os.makedirs('charts')

# Function to save and show charts
def save_and_show_chart(fig, filename):
    filepath = os.path.join('charts', filename)
    fig.savefig(filepath)
    st.image(filepath)

st.set_page_config(
    page_title="Breast Cancer Analysis",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )

# Set page title
st.title('Breast Cancer Diagnosis - Machine Learning Model Evaluation')
st.divider()

# Breast Cancer Wisconsin Dataset Information
st.header("Breast Cancer Wisconsin (Diagnostic) Data Set")
st.write("""
The **Breast Cancer Wisconsin (Diagnostic) Data Set** is a collection of clinical breast cancer diagnostic data. The data includes features that describe characteristics of cell nuclei from breast cancer biopsies, which are used to predict whether a tumor is benign or malignant.
Key features in the dataset include:
- **Radius:** The average distance from the center to points on the perimeter
- **Texture:** The standard deviation of grayscale values
- **Perimeter, Area, Smoothness:** Other morphological features describing cell shapes
The dataset is commonly used for classification tasks in machine learning to predict the likelihood of breast cancer malignancy.
""")

# Footer or additional information
st.write("This application aims to provide insights into breast cancer through data analysis and prediction based on the Breast Cancer Wisconsin dataset.")
st.divider()

# Data Preparation and Import Libraries
st.header('Data Preparation')
data = pd.read_csv('./data/data.csv')

st.subheader('Data Preview (show only first 5 rows)')
st.write(data.head())

st.subheader('Data Shape')
st.success(f"There are {data.shape[0]} rows and {data.shape[1]} columns in this dataset.")

# st.subheader('Data Info')
# st.table(data.info())
st.divider()
st.subheader('Checking Duplicates and Missing Values')
st.success(f"There are {data.duplicated().sum()} duplicate rows in this dataset.")
st.write(data.isna().sum())

st.divider()
st.subheader('Dropping Irrelevant Columns')
data = data.drop(['id', 'Unnamed: 32'], axis=1)
st.success("**id** and **Unnamed** columned were deleted because these variables can not be used for classification.")

st.divider()
st.subheader('Renaming Columns')
st.write("In the dataset, the 'diagnosis' variable was renamed as 'target,' where the value 'M' (Malignant) was renamed to 1 and 'B' (Benign) was renamed to 0 for easier modeling.")

data = data.rename(columns={'diagnosis': 'target'})
df = data.copy()
data.target.replace({'M': '1', 'B': '0'}, inplace=True)
data.target = data.target.astype('float64')

st.write(data.head())
st.divider()

# Analysis & EDA
st.header('Analysis & EDA')
st.subheader('Target Value Counts')

y = df.target
ax = sns.countplot(y,label="Count")       # M = 212, B = 357
B, M = y.value_counts()
st.write('Number of Benign (0): ',B)
st.write('Number of Malignant (1): ',M)

# Visualizing target data
st.subheader('Bar Plot of Target Values')
fig, ax = plt.subplots(figsize=(8, 6))
# data['target'].value_counts().plot(kind='bar', edgecolor='black', color=['lightsteelblue', 'navajowhite'], ax=ax)
sns.countplot(y, label="Count", ax=ax)
ax.set_title("Target Distribution")
# st.pyplot(fig)
save_and_show_chart(fig, 'target_count.png')
st.divider()

# Correlation Analysis
st.subheader('Correlation Analysis')
cor = data.corr()
st.write(cor)
st.divider()

st.subheader('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(25, 23))
sns.heatmap(cor, annot=True, linewidths=0.3, linecolor="black", fmt=".2f", ax=ax)
ax.set_title('Correlation Heatmap')
# st.pyplot(fig)
save_and_show_chart(fig, 'correlation_heatmap.png')
st.divider()

st.subheader('Features with Correlation > 0.75')
threshold = 0.75
filtre = np.abs(cor["target"]) > threshold
corr_features = cor.columns[filtre].tolist()

cluster_map = sns.clustermap(
    data[corr_features].corr(),
    annot=True,
    fmt=".2f",
    figsize=(10, 8)  # Adjust the figsize here
)
plt.savefig('charts/clustermap_high_correlation.png')
st.image('charts/clustermap_high_correlation.png')
# st.pyplot(cluster_map.fig)
st.divider()

# Pairplot for high-correlation features
st.subheader('Pairplot for Features with High Correlation')
fig, ax = plt.subplots(figsize=(8, 6))
pairplot = sns.pairplot(data[corr_features], diag_kind="kde", markers="*", hue="target")
plt.savefig('charts/pairplot_high_correlation.png')
st.image('charts/pairplot_high_correlation.png')
# st.pyplot(pairplot)

st.divider()
# Machine Learning Model Evaluation
st.header('Machine Learning Model Evaluation')

# Splitting the data
x = data.drop('target', axis=1)
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

algorithm = ['KNeighborsClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'LogisticRegression']
Accuracy = []

def evaluate_model(model):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    Accuracy.append(acc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    st.subheader(f'Confusion Matrix for {model.__class__.__name__}')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    save_and_show_chart(fig, f'{model}_confusion_matrix.png')
    # st.pyplot(fig)

    # Normalized Confusion Matrix
    cm_norm = confusion_matrix(y_test, pred, normalize='true')
    st.subheader(f'Normalized Confusion Matrix for {model.__class__.__name__}')
    fig, ax = plt.subplots()
    sns.heatmap(cm_norm, annot=True, cmap='Blues', ax=ax)
    save_and_show_chart(fig, f'{model}_normalized_confusion_matrix.png')
    # st.pyplot(fig)

    # Classification Report
    st.subheader(f'Classification Report for {model.__class__.__name__}')
    st.text(classification_report(y_test, pred))
    st.write(f"Accuracy: {acc}")

# Evaluating different models
st.subheader('01. KNeighborsClassifier Evaluation')
model_knn = KNeighborsClassifier(n_neighbors=2)
evaluate_model(model_knn)
st.divider()

st.subheader('02. RandomForestClassifier Evaluation')
model_rf = RandomForestClassifier(n_estimators=100, random_state=0)
evaluate_model(model_rf)
st.divider()

st.subheader('03. DecisionTreeClassifier Evaluation')
model_dt = DecisionTreeClassifier(random_state=42)
evaluate_model(model_dt)
st.divider()

st.subheader('04. GaussianNB Evaluation')
model_nb = GaussianNB()
evaluate_model(model_nb)
st.divider()

st.subheader('05. LogisticRegression Evaluation')
model_lr = LogisticRegression()
evaluate_model(model_lr)
st.divider()

# Final Accuracy Plot
st.header('Model Accuracy Comparison')
df = pd.DataFrame({'Algorithm': algorithm, 'Accuracy': Accuracy})

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(df.Algorithm, df.Accuracy, label='Accuracy', lw=5, color='peru', marker='o', markersize=15)
ax.legend(fontsize=15)
ax.set_xlabel('\nModel', fontsize=20)
ax.set_ylabel('Accuracy\n', fontsize=20)
save_and_show_chart(fig, 'model_Accuracy.png')
# st.pyplot(fig)

# End of Streamlit app
