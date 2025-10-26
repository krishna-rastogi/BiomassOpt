from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, 'templates'),
    static_folder=os.path.join(FRONTEND_DIR, 'static')
)

# Paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier_model.h5')
REGRESSOR_PATH = os.path.join(MODEL_DIR, 'regressor_model.h5')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')

# Load models
clf_model = load_model(CLASSIFIER_PATH, compile=False)
reg_model = load_model(REGRESSOR_PATH, compile=False)
preprocessor = joblib.load(PREPROCESSOR_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# ---------------- Routes ---------------- #

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model1')
def model1_page():
    return render_template('model1.html')

@app.route('/model2')
def model2_page():
    return render_template('model2.html')

@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        X_proc = preprocessor.transform(input_df)

        y_proba = clf_model.predict(X_proc)
        y_label = np.argmax(y_proba, axis=1)
        predicted_product = label_encoder.inverse_transform(y_label)[0]

        predicted_hhv = reg_model.predict(X_proc).squeeze()

        return jsonify({
            'predicted_product': predicted_product,
            'predicted_hhv': float(predicted_hhv)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ----------------------------------------- #

if __name__ == '__main__':
    app.run(debug=True)
