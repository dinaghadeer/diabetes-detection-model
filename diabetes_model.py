import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def load_data():
    df = pd.read_csv("diabetes.csv") 
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=42).fit(X_train, y_train),
        "Decision Tree": DecisionTreeClassifier(max_depth=4),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

def predict_user_input(models, scaler, user_input):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    predictions = {}
    for name, model in models.items():
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        predictions[name] = {"Prediction": pred, "Probability": round(prob, 2)}
    return predictions

# Streamlit app
st.title("Diabetes Prediction App")

# Load and train
df = load_data()
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler, X_train_scaled, X_test_scaled = preprocess(X_train, X_test)
models = train_models(X_train_scaled, y_train)

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
    results = predict_user_input(models, scaler, user_input)
    st.subheader("Model Predictions")
    for model_name, res in results.items():
        st.write(f"**{model_name}**: Prediction = {res['Prediction']}, Probability = {res['Probability']}")

    st.subheader("Model Comparison (on test data)")
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        st.write(f"**{name}** -> Accuracy: {acc:.2f}, AUC: {auc:.2f}")
