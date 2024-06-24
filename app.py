from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('DecisionTreeRegressor.pkl')  # Update with your model path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    temperature = int(request.form['Temperature'])
    humidity = int(request.form['Humidity'])
    windspeed = int(request.form['WindSpeed'])
    general_diffuse_flows = int(request.form['GeneralDiffuseFlows'])
    diffuse_flows = int(request.form['DiffuseFlows'])
    power_consumption_zone1 = int(request.form['PowerConsumption_Zone1'])
    power_consumption_zone2 = int(request.form['PowerConsumption_Zone2'])

    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[temperature, humidity, windspeed, general_diffuse_flows, diffuse_flows,
                                power_consumption_zone1, power_consumption_zone2]],
                              columns=['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows',
                                       'DiffuseFlows', 'PowerConsumption_Zone1', 'PowerConsumption_Zone2'])

    # Make a prediction using the model
    prediction = model.predict(input_data)[0]

    # Render the result page with the prediction
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
