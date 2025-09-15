import pickle
import streamlit as st
import numpy as np
import os
from sklearn.datasets import load_iris


st.title("Flower Classification App")

file_name = "model.pkl"
with open(os.path.join("model", file_name), "rb") as f:
    model = pickle.load(f)

sl = st.number_input("Insert a sepel length")
sw = st.number_input("Insert a sepel width")
pl = st.number_input("Insert a petal length")
pw = st.number_input("Insert a petal width")

if st.button("Predict"):

    sample = np.array([[sl, sw, pl, pw]])

    # If model supports predict_proba, use it
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)
        pred_index = np.argmax(probs)
        iris = load_iris()
        pred_class = iris.target_names[pred_index]
    else:
        pred = model.predict(sample)
        iris = load_iris()
        pred_class = iris.target_names[pred[0]]

    st.write("🔮 Predicted flower is:", pred_class)
