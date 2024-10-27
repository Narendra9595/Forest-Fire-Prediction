from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Define the exact feature order used during training so it ca be better understood by the model
FEATURE_NAMES = ['temperature', 'wind_speed', 'humidity', 'month', 'day', 
                 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

# Load the trained model 
with open('models/forest_fire_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values and perform calculations
        input_data = {
            'temperature': float(request.form['temperature']),
            'wind_speed': float(request.form['wind_speed']),
            'humidity': float(request.form['humidity']),
            'FFMC': float(request.form['FFMC']),
            'DMC': float(request.form['DMC']),
            'DC': float(request.form['DC']),
            'month': 6,  # Default value
            'day': 15,   # Default value
        }

        # Calculate derived indices
        input_data['ISI'] = input_data['FFMC'] * input_data['wind_speed'] / 100
        input_data['BUI'] = input_data['DMC'] * 0.8 + input_data['DC'] * 0.2
        input_data['FWI'] = input_data['ISI'] * 0.3 + input_data['BUI'] * 0.7

        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])[FEATURE_NAMES]

        # Make prediction
        probability = model.predict_proba(input_df)[0][1]

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
            risk_color = "success"
        elif probability < 0.7:
            risk_level = "Medium Risk"
            risk_color = "warning"
        else:
            risk_level = "High Risk"
            risk_color = "danger"

        # Calculate feature importance
        feature_importance = []
        for name, importance in zip(FEATURE_NAMES, model.feature_importances_):
            feature_importance.append({
                "name": name,
                "importance": f"{importance:.3f}"
            })

        # Prepare prediction results
        prediction_result = {
            "probability": f"{probability*100:.1f}",
            "risk_level": risk_level,
            "risk_color": risk_color,
            "explanations": feature_importance,
            "input_values": input_data
        }

        # Additional fire behavior metrics for the purpose of feature explanations
        prediction_result["fire_metrics"] = {
            "spread_rate": f"{input_data['ISI'] * 0.5:.2f} m/min",
            "intensity": f"{input_data['FWI'] * 100:.0f} kW/m",
            "flame_height": f"{(input_data['FWI'] * 0.3):.1f} m"
        }

        return render_template('index.html', 
                             prediction=prediction_result, 
                             show_results=True)

    except Exception as e:
        error_message = f"Prediction Error: {str(e)}"
        return render_template('index.html', error=error_message)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        input_data = request.get_json()
        # Process input and return JSON response
        input_df = pd.DataFrame([input_data])[FEATURE_NAMES]
        probability = model.predict_proba(input_df)[0][1]
        
        return {
            "success": True,
            "probability": probability,
            "risk_level": "High Risk" if probability > 0.7 else "Medium Risk" if probability > 0.3 else "Low Risk"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
