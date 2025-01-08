from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

app = Flask(__name__)

# Load and clean the dataset
data = pd.read_csv('Dataset.csv')

# Drop unnecessary columns
data_clean = data.drop(columns=[col for col in data.columns if 'Unnamed' in col])

# One-hot encode 'location' column
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Correct handling
location_encoded = pd.DataFrame(encoder.fit_transform(data_clean[['location']]), 
                                columns=encoder.get_feature_names_out(['location']))

# Combine the encoded 'location' with the dataset
data_clean = pd.concat([data_clean.drop(columns=['location']), location_encoded], axis=1)

# Define features and target
features = ['beds', 'baths', 'size', ] + list(location_encoded.columns)
X = data_clean[features]
y = data_clean['price']

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the user
        beds = float(request.form['beds'])
        baths = float(request.form['baths'])
        size = float(request.form['size'])
        location = request.form['location']

        # One-hot encode the location input
        location_encoded = encoder.transform([[location]])  # No need for .toarray()

        # Prepare the input data for prediction
        input_data = np.array([[beds, baths, size, ] + list(location_encoded[0])])

        # Predict the price
        predicted_price = model.predict(input_data)[0]
        return jsonify({'predicted_price': round(float(predicted_price), 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
