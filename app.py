from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import save_object
from sklearn.preprocessing import StandardScaler

appplication=Flask(__name__)

app=appplication

# Route for a home page

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=['Get',"Post"])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.get("test_preperation_course"),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get("writing_score"))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)


        Predict_pipeline=PredictPipeline()
        results = Predict_pipeline.predict(pred_df)
        return render_template("home.html",results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)