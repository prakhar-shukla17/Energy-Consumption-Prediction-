from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collecting the Datetime input from the form
        datetime_input = request.form.get('datetime')
        
        # Creating a DataFrame with the input
        data = CustomData(datetime=datetime_input)
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:")
        print(pred_df)
        print("Before Prediction")

        # Initializing the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Returning the prediction result to the user
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)