import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# --- Load the model and columns ONCE when the app starts up ---
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
except Exception as e:
    model = None
    model_columns = None
    print(f"CRITICAL ERROR: Could not load model files. {e}")

# --- Define the prediction endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_columns is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    # Get the JSON data sent from PHP
    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'Invalid JSON format'}), 400

    try:
        # Create a one-row DataFrame
        df = pd.DataFrame([json_data])

        # Preprocess the data EXACTLY like the training data
        df_encoded = pd.get_dummies(df, columns=['Barangay', 'Classification'])
        df_aligned = df_encoded.reindex(columns=model_columns, fill_value=0)

        # Make the prediction
        prediction = model.predict(df_aligned)
        
        # Return the result as JSON
        return jsonify({'predicted_collection': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # This part is for local testing, not used by Render
    app.run(debug=True)