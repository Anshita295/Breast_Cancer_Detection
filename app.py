import streamlit as st
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Prepare data
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, Y_train)

# Save model
joblib.dump(model, 'breast_cancer_model.pkl')

# Load model
model = joblib.load("breast_cancer_model.pkl")

# Streamlit UI
st.title("🔬 Breast Cancer Prediction App")
st.write("Enter values below to predict if the tumor is **Malignant** or **Benign**.")

# User Input Form
features = []
for feature_name in breast_cancer_dataset.feature_names:
    value = st.number_input(f"{feature_name}", step=0.01)
    features.append(value)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    result = "Malignant" if prediction[0] == 0 else "Benign"
    st.success(f"🔔 Prediction: The tumor is **{result}**")
