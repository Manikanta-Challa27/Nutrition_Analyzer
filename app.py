import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('model/nutrition_model.pkl')

# Get model's expected features (ensure your sklearn version supports this)
try:
    model_features = model.feature_names_in_
except AttributeError:
    # Fallback: manually define features if model doesn't store them
    model_features = ['fat', 'protein', 'carbohydrate', 'sugar', 'fiber', 'sodium']  # Adjust if needed

# Load the nutrition dataset
df = pd.read_excel("data/nutrition.xlsx")

# Clean values
def clean_value(val):
    try:
        return float(str(val).replace('g', '').replace('mg', '').replace('kcal', '').strip())
    except:
        return 0

target_col = 'calories'
nutrient_cols = [col for col in model_features if col in df.columns]

# Clean the dataframe
for col in nutrient_cols + [target_col]:
    if col in df.columns:
        df[col] = df[col].apply(clean_value)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        if not data or 'age' not in data or 'food_items' not in data:
            raise ValueError("Missing 'age' or 'food_items' in request data.")

        age = int(data['age'])
        food_items = [item.lower().strip() for item in data['food_items']]

        total_calories = 0
        total_nutrients = {nutrient: 0 for nutrient in nutrient_cols}
        not_found = []

        for food in food_items:
            match = df[df['name'].str.lower().str.contains(food)]
            if not match.empty:
                row = match.iloc[0]
                nutrient_input = pd.DataFrame([[row[n] for n in nutrient_cols]], columns=nutrient_cols)
                predicted_cal = model.predict(nutrient_input)[0]
                total_calories += predicted_cal
                for n in nutrient_cols:
                    total_nutrients[n] += row[n]
            else:
                not_found.append(food)

        # Age-based recommendations
        if age < 18:
            recommended = {
                'fat': (50, 70),
                'protein': (40, 52),
                'carbohydrate': (180, 260),
                'fiber': (20, 30.4),
                'sodium': (1200, 1840)
            }
        else:
            recommended = {
                'fat': (60, 80),
                'protein': (50, 65),
                'carbohydrate': (220, 300),
                'fiber': (25, 35),
                'sodium': (1500, 2300)
            }

        recommendations = {}
        for nutrient, value in total_nutrients.items():
            min_val, max_val = recommended.get(nutrient, (0, 0))
            if value < min_val:
                status = 'low'
            elif value > max_val:
                status = 'high'
            else:
                status = 'good'
            recommendations[nutrient] = {
                'value': round(value, 2),
                'status': status,
                'recommended_range': (min_val, max_val)
            }

        return jsonify({
            'age': age,
            'total_calories': round(total_calories, 2),
            'total_nutrients': total_nutrients,
            'recommendations': recommendations,
            'not_found_items': not_found
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
