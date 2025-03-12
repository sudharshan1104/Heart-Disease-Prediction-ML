from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the frontend (the form fields)
    data = request.get_json()

    # Convert data to pandas DataFrame
    input_data = pd.DataFrame([data])

    # Load the pre-trained model (ensure 'best_model.pkl' is in the same directory)
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Make the prediction using the model
    prediction = model.predict(input_data)

    # Convert numerical prediction to categorical prediction
    if prediction[0] < 0.33:
        risk_category = 'Low Risk'
    elif prediction[0] < 0.66:
        risk_category = 'Moderate Risk'
    else:
        risk_category = 'High Risk'

    # Return the categorical prediction as JSON
    return jsonify({'prediction': risk_category})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Running on port 5000
