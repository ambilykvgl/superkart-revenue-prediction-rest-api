# Import necessary libraries
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_predictor_api = Flask("SuperKart Revenue Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@superkart_predictor_api.get('/')
def home():
    return "Welcome to the SuperKart Revenue Prediction API!"

# Define an endpoint for single revenue prediction (POST request)
@superkart_predictor_api.post('/v1/predict')
def predict_revenue():
    product_data = request.get_json()

    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Store_Age_Years': product_data['Store_Age_Years'],
        'Product_Type_Category': product_data['Product_Type_Category'],
        'Product_Id_char': product_data['Product_Id_char']
    }

    input_data = pd.DataFrame([sample])

    prediction = model.predict(input_data)[0]
    predicted_revenue = round(float(prediction), 2)

    return jsonify({'Predicted Revenue (in dollars)': predicted_revenue})

# Define an endpoint for batch prediction (POST request)
@superkart_predictor_api.post('/v1/predictbatch')
def predict_revenue_batch():

   # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    expected_cols = [
        'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_MRP', 'Store_Size', 'Store_Location_City_Type',
        'Store_Type', 'Store_Age_Years', 'Product_Type_Category',
        'Product_Id_char'
    ]

    input_data = input_data[expected_cols]

    predictions = model.predict(input_data).tolist()
    predicted_revenues = [round(float(val), 2) for val in predictions]

    output_dict = dict(zip(range(len(predicted_revenues)), predicted_revenues))
    return jsonify(output_dict)

# Run the Flask application
if __name__ == '__main__':
    superkart_predictor_api.run(debug=True)
