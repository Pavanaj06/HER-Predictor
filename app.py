import gradio as gr
import pandas as pd
import joblib

# Load trained model
model = joblib.load("HER_XGBoost.pkl")


def predict_her(
    steel,
    type_,
    thickness,
    ys,
    uts,
    n,
    clearance,
    hole,
    burr,
    punch,
):

    # Feature engineering
    strength_ratio = ys / uts
    uts_n = uts * n

    # Create dataframe
    sample = pd.DataFrame({
        "Source": ["User"],
        "Steel": [steel],
        "Type": [type_],
        "Thickness_mm": [thickness],
        "YS_MPa": [ys],
        "UTS_MPa": [uts],
        "n_value": [n],
        "Clearance_pct": [clearance],
        "Hole_Preparation": [hole],
        "Burr_Orientation": [burr],
        "Punch_Geometry": [punch],
        "Strength_Ratio_YS_UTS": [strength_ratio],
        "UTS_x_n": [uts_n]
    })

    # Prediction
    her = model.predict(sample)[0]

    # Rating
    if her < 40:
        quality = "🔴 Poor Stretch Flangeability"
    elif her < 70:
        quality = "🟡 Moderate Stretch Flangeability"
    else:
        quality = "🟢 Excellent Stretch Flangeability"

    return round(her, 2), quality


demo = gr.Interface(
    fn=predict_her,

    inputs=[
        gr.Dropdown(
            ["DP590", "DP600", "DP780", "DP980",
             "CP590", "CP780", "CP980"],
            label="Steel"
        ),

        gr.Dropdown(
            ["DP", "CP"],
            label="Type"
        ),

        gr.Number(label="Thickness (mm)", value=1.4),

        gr.Number(label="Yield Strength (MPa)", value=600),

        gr.Number(label="Ultimate Tensile Strength (MPa)", value=980),

        gr.Number(label="n-value", value=0.12),

        gr.Number(label="Clearance (%)", value=12),

        gr.Dropdown(
            ["Sheared", "Reamed", "EDM"],
            label="Hole Preparation"
        ),

        gr.Dropdown(
            ["Up", "Down"],
            label="Burr Orientation"
        ),

        gr.Dropdown(
            ["Flat", "Conical"],
            label="Punch Geometry"
        )

    ],

    outputs=[
        gr.Number(label="Predicted HER (%)"),
        gr.Textbox(label="Quality")
    ],

    title="🔩 Hole Expansion Ratio Predictor",

    description="Machine Learning based HER Prediction using XGBoost"
)

demo.launch()