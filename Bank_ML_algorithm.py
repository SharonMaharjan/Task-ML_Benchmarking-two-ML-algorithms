import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import streamlit as st

# Load the dataset
bank_data = pd.read_csv("Data/bank-full.csv", sep=';', quotechar='"')

# Create a sidebar for user options
st.sidebar.header("Bank Dataset Analysis Options")
show_data = st.sidebar.checkbox("Show Dataset")
show_statistics = st.sidebar.checkbox("Show Dataset Statistics")
show_missing = st.sidebar.checkbox("Show Missing Values")
show_column_names = st.sidebar.checkbox("Show Column Names")

# Display the first and last few rows
if show_data:
    st.header("Bank Marketing Dataset")
    st.subheader("First few rows of the dataset:")
    st.dataframe(bank_data.head())
    st.subheader("Last few rows of the dataset:")
    st.dataframe(bank_data.tail())

# Display dataset statistics and check for missing values
if show_statistics:
    st.subheader("Summary statistics of the dataset:")
    st.write(bank_data.describe())

if show_missing:
    st.subheader("Missing values in the dataset:")
    st.write(bank_data.isnull().sum())

# Display column names
if show_column_names:
    st.subheader("Column names:")
    st.write(bank_data.columns)

# Select features and target variable
feature_cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                'previous', 'poutcome']
X = bank_data[feature_cols]
y = bank_data['y']

# Perform one-hot encoding
X = pd.get_dummies(X, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Decision Tree Classifier
bank_decision_tree = DecisionTreeClassifier(criterion="entropy")
bank_decision_tree = bank_decision_tree.fit(X_train, y_train)
y_pred = bank_decision_tree.predict(X_test)
decision_tree_accuracy = metrics.accuracy_score(y_test, y_pred)

# K-Nearest Neighbors Classifier
bank_data_k = KNeighborsClassifier(n_neighbors=5)
bank_data_k.fit(X_train, y_train)
y_pred_k = bank_data_k.predict(X_test)
knn_accuracy = metrics.accuracy_score(y_test, y_pred_k)

# Confusion Matrix for Decision Tree
confusion_decision_tree = confusion_matrix(y_test, y_pred)

# Streamlit app
st.title("Bank Dataset Analysis")

st.write("### Decision Tree Classifier")
st.write(f"Accuracy (Decision Tree): {decision_tree_accuracy:.4f}")

st.write("### K-Nearest Neighbors Classifier")
st.write(f"Accuracy (K-Nearest Neighbors): {knn_accuracy:.4f}")

st.write("### Confusion Matrix (Decision Tree)")
st.dataframe(confusion_decision_tree)

st.write("### Confusion Matrix Heatmap (Decision Tree)")
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_decision_tree, annot=True, fmt="d", cmap="Blues")
st.pyplot()
