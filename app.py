import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Support Vector Machine (SVM)": SVC(probability=True, kernel="rbf", C=2, gamma=0.1, random_state=42),
}

st.sidebar.title("⚙️ Model Selection")
model_choice = st.sidebar.selectbox("Choose Classifier", list(models.keys()))

model = models[model_choice]
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("🔮 Diabetes Risk Prediction System")
st.write("Using Pima Indian Diabetes Dataset with Multiple Models")

st.sidebar.header("📌 Enter Patient Data")

def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("📊 Patient Data Entered")
st.write(input_df)

st.subheader("📈 Model Accuracy on Test Data")
st.write(f"{accuracy*100:.2f}% ({model_choice})")

st.subheader("🔍 Prediction")
st.write("🩸 Diabetic" if prediction[0] == 1 else "✅ Not Diabetic")

st.subheader("📊 Prediction Probability")
st.write(f"Not Diabetic: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Diabetic: {prediction_proba[0][1]*100:.2f}%")
