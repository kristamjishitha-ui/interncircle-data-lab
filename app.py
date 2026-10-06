import streamlit as st
import joblib
import numpy as np

model = joblib.load("iris_decision_tree_model.pkl")

class_names = ["Setosa", "Versicolor", "Virginica"]

st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower measurements to predict its species.")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0, 0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0, 0.1)

if st.button("Predict Iris Species"):
    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)[0]
    predicted_species = class_names[prediction]

    st.success(f"🌸 Predicted Species: {predicted_species}")