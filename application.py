from flask import Flask, render_template, request
import joblib
import numpy as np
import joblib

app = Flask(__name__)
# scaler = pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    gender = (request.form['gender'])
    age = (request.form['age'])
    hypertension = (request.form['hypertension'])
    heart_disease = (request.form['heart_disease'])
    ever_married = (request.form['ever_married'])
    work_type = (request.form['work_type'])
    Residence_type = (request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = (request.form['smoking_status'])

    x=np.array([gender, age, hypertension, heart_disease, ever_married, work_type,
                Residence_type, avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

  
    # x = scaler.transform(x)
    svc = joblib.load("stroke.joblib") 
    y_pred = svc.predict(x)

    # if No Stroke Risk
    if y_pred == 0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)


