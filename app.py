from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
app = Flask(__name__)
#<img src="/static/khphoto.jpg" width="100" height="100">
# Load the trained LinearRegression model
model = joblib.load('LReg_model1.joblib')
#model = joblib.load('KNN_model1.joblib')
#model = joblib.load('nb_model1.joblib')
#model = joblib.load('lg_model1.joblib')
#model = joblib.load('svm_model1.joblib')

print("Model loaded successfully")
Mean_squared_Error = joblib.load('rmse_1.joblib')
Mean_Absolute_Error = joblib.load('MASR.joblib')
R_Score2 = joblib.load('MASR1.joblib')

# Define ranges for user inputs
input_ranges = {
    
    'failures': (0,4), #number of past class failures (numeric: n if 1<=n<3, else 4)
    'Medu': (0, 4),   #mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
    'higher': (0, 1), #wants to take higher education (binary: yes or no)
    'age': (15, 22), # student's age (numeric: from 15 to 22)
    'Fedu': (0, 4),  # father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
    'goout': (1, 5), # going out with friends (numeric: from 1 - very low to 5 - very high)
    'romantic': (0, 1), # with a romantic relationship (binary: yes or no)
    'reason': (1, 4)  #reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
}
@app.route('/')
def home():
    # Load the preprocessed data from the CSV file
    preprocessed_data = pd.read_csv('preprocessed_data.csv')

    # Convert the preprocessed data to a list of dictionaries
    preprocessed_data_list = preprocessed_data.to_dict(orient='records')

    return render_template('index.html', preprocessed_data=preprocessed_data_list, mean_absolute_error=Mean_Absolute_Error,
                           mean_square_error=Mean_squared_Error, R2_score=R_Score2)
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    failures = int(request.form['failures'])  # Convert failures to integer
    Medu = int(request.form['Medu'])  # Convert Medu to integer
    higher = int(request.form['higher'])  # Convert higher to integer
    age = int(request.form['age'])  # Convert age to integer
    Fedu = int(request.form['Fedu'])  # Convert Fedu to integer
    goout = int(request.form['goout'])  # Convert goout to integer
    romantic = int(request.form['romantic'])  # Convert romantic to integer
    reason = int(request.form['reason'])  # Convert reason to integer
    
    # Make prediction
    prediction = model.predict([[failures, Medu, higher, age, Fedu, goout,romantic,reason]])
    
    # Classify the prediction
    if prediction < 10:
        prediction = 'Fail'
    else:
        prediction = 'Good'
    
    # Return the prediction
    return jsonify({'prediction': prediction, 'mean_absolute_error': Mean_Absolute_Error,
                    'mean_square_error':Mean_squared_Error,
                    'R2_score':R_Score2,})
    
if __name__ == '__main__':
    app.run(debug=True)