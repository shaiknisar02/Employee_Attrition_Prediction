from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model, feature columns, encoders, and scaler
with open("attrition_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    error = ""
    categorical_options = {}

    # Prepare categorical options for dropdowns
    for feature, encoder in le_dict.items():
        categorical_options[feature] = list(encoder.classes_)

    if request.method == "POST":
        try:
            input_data = {}
            for feature in feature_columns:
                value = request.form.get(feature)

                # Encode categorical values
                if feature in le_dict:
                    le = le_dict[feature]
                    if value in le.classes_:
                        value = le.transform([value])[0]
                    else:
                        raise ValueError(f"Unknown value '{value}' for feature '{feature}'")
                else:
                    value = float(value)

                input_data[feature] = value

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            result = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]

            prediction = f"⚠️ Yes - This employee is likely to leave (Probability: {prob:.2f})" if result == 1 \
                         else f"✅ No - This employee is likely to stay (Probability: {prob:.2f})"

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error,
                           features=feature_columns, categorical_options=categorical_options)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
