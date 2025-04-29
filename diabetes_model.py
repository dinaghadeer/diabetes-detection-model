import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv("diabetes.csv") 
    cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero_invalid] = df[cols_with_zero_invalid].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def train_models(X_train, y_train, X_test, y_test, scaler):
    models = {
        "Logistic Regression": LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = accuracy_score(y_test, y_pred)
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]

    # Save best model and scaler
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    return best_model_name, best_model

def predict_user_input(model, scaler, user_input):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    result = "Diabetic" if pred == 1 else "Not Diabetic"
    return result, round(prob, 2)

# Streamlit app
st.title("Diabetes Prediction App")

# Load or train model
if os.path.exists("best_model.pkl") and os.path.exists("scaler.pkl"):
    best_model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    best_model_name = type(best_model).__name__.replace("Classifier", "")
else:
    df = load_data()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler, X_train_scaled, X_test_scaled = preprocess(X_train, X_test)
    best_model_name, best_model = train_models(X_train_scaled, y_train, X_test_scaled, y_test, scaler)

# User input sidebar
st.sidebar.header("User Input Features")
user_input = {
    "Pregnancies": st.sidebar.slider("Pregnancies", 0, 17, 3),
    "Glucose": st.sidebar.slider("Glucose", 50, 200, 117),
    "BloodPressure": st.sidebar.slider("BloodPressure", 30, 122, 72),
    "SkinThickness": st.sidebar.slider("SkinThickness", 0, 99, 23),
    "Insulin": st.sidebar.slider("Insulin", 0, 846, 30),
    "BMI": st.sidebar.slider("BMI", 10.0, 67.1, 32.0),
    "DiabetesPedigreeFunction": st.sidebar.slider("DiabetesPedigreeFunction", 0.05, 2.42, 0.3725),
    "Age": st.sidebar.slider("Age", 21, 81, 29)
}

if st.button("Predict"):
    result, prob = predict_user_input(best_model, scaler, user_input)
    st.subheader(f"Prediction Result Using {best_model_name}")

    if result == "Diabetic":
        st.error(f"You are: {result} (Probability: {prob})")
    else:
        st.success(f"You are: {result} (Probability: {prob})")