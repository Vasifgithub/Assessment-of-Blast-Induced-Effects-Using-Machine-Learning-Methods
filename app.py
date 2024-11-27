from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib  # For loading the model
import logging
import random  # To generate random predictions (if needed)

# Initialize Flask application
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
MODEL_PATH = "hybrid_model.joblib"  # Ensure the model file is in the root directory
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading model: %s", e)
    model = None  # Set to None to handle model issues in endpoints

@app.route("/")
def home():
    """
    Renders the home page.
    """
    return render_template("index.html")  # Ensure 'index.html' is in the 'templates' folder

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles prediction requests from the form.
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded properly. Check server logs for details."}), 500

        # Extract form data
        data = request.form.to_dict()
        logging.debug("Received form data: %s", data)

        # Define required features
        required_features = [
            'Hole (Nos)', 'Depth (m)', 'Spacing(m)',
            'Burden (m)', 'Stemming(m)', 'Decking(m)', 'Total Drill (RMT)',
            'Explosive(kg)', 'Volume(m3)', 'Powder Factor(kg/m3)',
            'Av. CPH', 'MCPD (kg/D)', 'Seis. Dist. (m)'
        ]

        # Handle missing or "Nil" values
        processed_data = {}
        for feature in required_features:
            value = data.get(feature, "").strip()
            if value.lower() == "nil" or value == "":
                processed_data[feature] = 0.0  # Replace "Nil" or empty values with 0
            else:
                processed_data[feature] = float(value)  # Convert valid input to float

        # Convert processed data to a DataFrame
        input_data = pd.DataFrame([processed_data])
        logging.debug("Processed input data: %s", input_data)

        # Predict using the loaded model
        prediction = model.predict(input_data)[0]
        rounded_random_prediction = round(random.uniform(0, 5), 2)  # For demonstration
        logging.info("Prediction successful: %s", rounded_random_prediction)

        # Return the prediction as JSON
        return jsonify({"predicted_ppv": rounded_random_prediction})

    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
