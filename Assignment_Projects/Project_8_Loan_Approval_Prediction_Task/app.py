from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('best_loan_approval_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the preprocessing pipeline (same as in training)
categorical_features = ['education', 'self_employed']
numerical_features = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

# Create transformers with specific categories to ensure consistent feature count
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(
    categories=[['Graduate', 'Not Graduate'], ['No', 'Yes']],  # Specify exact categories
    handle_unknown='ignore'
)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        # Get form data
        data = request.form.to_dict()
        
        # Convert to appropriate data types
        input_data = {
            'no_of_dependents': int(data['no_of_dependents']),
            'education': data['education'],
            'self_employed': data['self_employed'],
            'income_annum': float(data['income_annum']),
            'loan_amount': float(data['loan_amount']),
            'loan_term': int(data['loan_term']),
            'cibil_score': int(data['cibil_score']),
            'residential_assets_value': float(data['residential_assets_value']),
            'commercial_assets_value': float(data['commercial_assets_value']),
            'luxury_assets_value': float(data['luxury_assets_value']),
            'bank_asset_value': float(data['bank_asset_value'])
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocess the data (use transform instead of fit_transform to avoid refitting)
        # We need to fit the preprocessor first with sample data to get the right structure
        if not hasattr(preprocessor, 'fitted_'):
            # Create sample data to fit the preprocessor once
            sample_data = pd.DataFrame([{
                'no_of_dependents': 2,
                'education': 'Graduate',
                'self_employed': 'No',
                'income_annum': 5000000,
                'loan_amount': 15000000,
                'loan_term': 12,
                'cibil_score': 750,
                'residential_assets_value': 2000000,
                'commercial_assets_value': 1000000,
                'luxury_assets_value': 500000,
                'bank_asset_value': 1000000
            }])
            preprocessor.fit(sample_data)
            preprocessor.fitted_ = True
        
        processed_data = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Get confidence score
        confidence = max(probability) * 100
        
        # Convert prediction to readable format
        result = "Approved" if prediction == 1 else "Rejected"
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 2),
            'probability_approved': round(probability[1] * 100, 2),
            'probability_rejected': round(probability[0] * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
