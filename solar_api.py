from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# === Model Training ===
df = pd.read_csv("Data 1 - Solar Effeciency - Hourly(2020-01-01)(2024-7-30).csv")
X = df[['MO', 'DY', 'RH2M', 'SZA', 'CLRSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DWN', 'T2M', 'PRECTOTCORR']]
y = df[['Effeciency', 'Power Output']]

model = RandomForestRegressor()
model.fit(X, y)

# === Flask App ===
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = [
            data['month'],
            data['day'],
            data['rh2m'],
            data['sza'],
            data['clrs'],
            data['alls'],
            data['temp'],
            data['precip']
        ]
        df_input = pd.DataFrame([input_data], columns=X.columns)
        result = model.predict(df_input)[0]
        return jsonify({
            "efficiency": round(result[0], 4),
            "power_output": round(result[1], 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
