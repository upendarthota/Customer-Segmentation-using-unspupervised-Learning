from flask import Flask, request, jsonify,render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is inside a 'templates' folder

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from input
    age = data['Age']
    annual_income = data['AnnualIncome']
    spending_score = data['SpendingScore']

    # Scale the input data
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(np.array([[age, annual_income, spending_score]]))

    # Make predictions
    cluster = kmeans_model.predict(scaled_input)
    pca_result = pca_model.transform(scaled_input)

    # Return the prediction results
    return jsonify({
        'Cluster': int(cluster[0]),
        'PCA_X': pca_result[0][0],
        'PCA_Y': pca_result[0][1]
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
