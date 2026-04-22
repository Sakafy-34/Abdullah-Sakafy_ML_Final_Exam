

import gradio as gr
import pandas as pd
import pickle
import numpy as np


with open("gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_charge(age, sex, bmi, children, smoker, region):
    
    # Pack inputs into a DataFrame
    # The column names must match your CSV file exactly
    input_df = pd.DataFrame([[
    age, sex, bmi, children, smoker, region
    ]],
    columns=[
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region"
    ]
    )
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    
    return f"Predicted Insurance Charge: ${prediction:.2f}"

# 3. The App Interface
# Defining inputs in a list to keep it clean
inputs = [
    
    gr.Number(label="Age", value=18),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Number(label="BMI", value=25),
    gr.Slider(0, 5, step=1,label="Children"),
    gr.Radio(['yes', 'no'], label="Smoker"),
    gr.Radio(['southwest', 'southeast', 'northwest', 'northeast'], label="Region")
]


app = gr.Interface(
    fn=predict_charge,
    inputs=inputs,
    outputs="text", 
    title="Insurance Charge Predictor")

app.launch(share=True)

