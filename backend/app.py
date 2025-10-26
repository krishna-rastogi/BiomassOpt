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


# ---------------- Model3 Setup ---------------- #
# Go up one level from backend, then enter data folder
DISTRICTS_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'districts_bounding_boxes.csv'))

# Load CSV
df_districts = pd.read_csv(DISTRICTS_PATH)
df_districts['center_lat'] = (df_districts['min_lat'] + df_districts['max_lat']) / 2
df_districts['center_lon'] = (df_districts['min_lon'] + df_districts['max_lon']) / 2


def calculate_distance_to_districts(lat, lon, df):
    distances = []
    for _, row in df.iterrows():
        district_lat = row['center_lat']
        district_lon = row['center_lon']
        distance = np.sqrt((lat - district_lat)**2 + (lon - district_lon)**2)
        distances.append(distance)
    return distances

def find_closest_district(lon, lat, df):
    if df.empty:
        return "No districts found", "No districts found"
    distances = calculate_distance_to_districts(lat, lon, df)
    min_distance_index = distances.index(min(distances))
    closest_row = df.iloc[min_distance_index]
    return closest_row['state'], closest_row['district']

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


# ---------------- Model3 Routes ---------------- #

@app.route('/model3')
def model3_page():
    return render_template('model3.html')

@app.route('/predict_nearest_city', methods=['POST'])
def predict_nearest_city():
    try:
        data = request.get_json()
        lon = float(data.get('longitude'))
        lat = float(data.get('latitude'))

        state, district = find_closest_district(lon, lat, df_districts)

        return jsonify({
            'state': state,
            'district': district
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    
# ----------------------------------------- #

if __name__ == '__main__':
    app.run(debug=True)
