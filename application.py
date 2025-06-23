from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import joblib  # ✅ Use joblib instead of pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Simplified CORS usage

# ✅ Load model using joblib
model = joblib.load('LinearRegressionModel.pkl')
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=years,
                           fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Get form data
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))

    # ✅ Create a DataFrame for prediction
    input_df = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=[[car_model, company, year, driven, fuel_type]])

    # ✅ Predict using the model
    prediction = model.predict(input_df)
    print(f"Prediction: {prediction}")

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)
