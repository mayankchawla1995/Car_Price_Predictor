from flask import Flask, render_template, request
import pandas as pd
import pickle
import sklearn


app=Flask(__name__)

model = pickle.load(open("LinarRegressionModel.pkl", 'rb'))
car=pd.read_csv("Cleaned_car_data.csv")


@app.route('/') #entry point for our application
def index():
    companies = sorted(car['Company'].unique())
    car_models = sorted(car['Name'].unique())
    year = sorted(car['Year'].unique(), reverse=True)
    fuel_type = car['Fuel_type'].unique()
    owner = sorted(car['Owner'].unique())
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type, owners =owner)
@app.route('/predict',methods=['POST'])
def predict():
    company= request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    owner = request.form.get('owner')

    prediction = model.predict(pd.DataFrame([[car_model, kms_driven, fuel_type, year, company, owner]], columns=['Name','Kms_driven', 'Fuel_type', 'Year', 'Company', 'Owner']))

    return str(prediction[0])



if __name__=="__main__":
    app.run(debug=True)