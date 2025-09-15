from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Load trained model
pipe = joblib.load("pipe.joblib")

app = Flask(__name__)

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Dharamsala', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Cuttack', 'Visakhapatnam',
          'Mohali', 'Bengaluru']

@app.route("/")
def index():
    return render_template("index.html", teams=teams, cities=cities)

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    batting_team = request.form.get("batting_team")
    bowling_team = request.form.get("bowling_team")
    city = request.form.get("city")
    
    try:
        target = int(request.form.get("target"))
        score = int(request.form.get("score"))
        overs = float(request.form.get("overs"))
        wickets_left = int(request.form.get("wickets_left"))
    except (TypeError, ValueError):
        return "Error: Invalid numeric input. Please enter all required fields."

    # Calculate derived features
    runs_left = target - score
    balls_left = (20 - overs) * 6  # assuming T20 match
    current_run_rate = score / overs if overs > 0 else 0
    required_run_rate = runs_left / (balls_left / 6) if balls_left > 0 else 0
    total_runs_x = target + 1  # assuming target = total runs

    # Build input dataframe
    input_df = pd.DataFrame([{
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'city': city,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_left': wickets_left,
        'total_runs_x': total_runs_x,
        'current_run_rate': current_run_rate,
        'required_run_rate': required_run_rate
    }])

    # Predict
    result = pipe.predict_proba(input_df)[0]
    prediction = f"Win Probability: {round(result[1] * 100, 2)}% | Lose Probability: {round(result[0] * 100, 2)}%"

    return render_template("index.html", teams=teams, cities=cities, prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)