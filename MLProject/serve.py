import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, jsonify

app = Flask(__name__)

# Muat model dengan run_id yang benar
model = mlflow.pyfunc.load_model("runs:/36cb9a86397340d7b8af4897fc55a54d/model")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)