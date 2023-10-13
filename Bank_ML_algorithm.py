
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

st.title("Bank Marketing Streamlit App")

# Load the data
@st.cache
def load_data():
    bank_data = pd.read_csv("resources/bank-full.csv", sep=';', quotechar='"')
    return bank_data

def main():
    bank_data = load_data()

    st.header("Exploratory Data Analysis (EDA)")
    st.dataframe(bank_data.head())

    st.subheader("Data Summary")
    st.write(bank_data.describe())

    st.subheader("Missing Values")
    st.write(bank_data.isnull().sum())

    st.subheader("Column Names")
    st.write(bank_data.columns)

    feature_cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                    'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                    'previous', 'poutcome']

    X = bank_data[feature_cols]

    y = bank_data['y']

    X = pd.get_dummies(X, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.header("Machine Learning Models")

    model_option = st.selectbox("Select a Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

    if model_option == "Logistic Regression":
        model = LogisticRegression()
    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier(criterion="entropy")
    elif model_option == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()