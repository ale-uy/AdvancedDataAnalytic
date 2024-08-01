from flask import Flask, render_template, request, redirect, url_for
import xgboost as xgb
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model with joblib
model = joblib.load('model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        satisfaction_level = float(request.form['satisfaction_level'])
        last_evaluation = float(request.form['last_evaluation'])
        number_project = int(request.form['number_project'])
        average_monthly_hours = int(request.form['average_monthly_hours'])
        time_spend_company = int(request.form['time_spend_company'])
        work_accident_1 = int(request.form['work_accident_1'])
        promotion_last_5years_1 = int(request.form['promotion_last_5years_1'])
        department = request.form['department']
        salary = request.form['salary']

        # Print form data for debugging
        print(f"Received form data: satisfaction_level={satisfaction_level}, last_evaluation={last_evaluation}, number_project={number_project}, average_monthly_hours={average_monthly_hours}, time_spend_company={time_spend_company}, work_accident_1={work_accident_1}, promotion_last_5years_1={promotion_last_5years_1}, department={department}, salary={salary}")

        # Create dictionary for departments
        departments = {
            'RandD': 0, 'accounting': 0, 'hr': 0, 'management': 0,
            'marketing': 0, 'product_mng': 0, 'sales': 0, 'support': 0, 'technical': 0
        }
        departments[department] = 1

        # Create dictionary for salary
        salaries = {'low': 0, 'medium': 0}
        salaries[salary] = 1

        # Create DataFrame with the data
        data = pd.DataFrame({
            'satisfaction_level': [satisfaction_level],
            'last_evaluation': [last_evaluation],
            'number_project': [number_project],
            'average_monthly_hours': [average_monthly_hours],
            'time_spend_company': [time_spend_company],
            'work_accident_1': [work_accident_1],
            'promotion_last_5years_1': [promotion_last_5years_1],
            'department_RandD': [departments['RandD']],
            'department_accounting': [departments['accounting']],
            'department_hr': [departments['hr']],
            'department_management': [departments['management']],
            'department_marketing': [departments['marketing']],
            'department_product_mng': [departments['product_mng']],
            'department_sales': [departments['sales']],
            'department_support': [departments['support']],
            'department_technical': [departments['technical']],
            'salary_low': [salaries['low']],
            'salary_medium': [salaries['medium']]
        })

        # Print DataFrame for debugging
        print(f"DataFrame:\n{data}")

                # Make prediction
        prediction = model.predict(data)[0]

        # Get probability
        probability = model.predict_proba(data)[0][1]

        # Redirect to results page
        return redirect(url_for('result', prediction=int(prediction), probability=float(probability)))

    return render_template('form.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction', type=int)
    probability = request.args.get('probability', type=float)
    return render_template('result.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
