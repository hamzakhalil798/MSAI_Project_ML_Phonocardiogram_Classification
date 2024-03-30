from flask import Flask, render_template, request,flash

import os
import numpy as np
from PIL import Image
from io import BytesIO
import joblib
import pickle
import warnings
from resources.utils import feature_1,feature_2,feature_3
import pandas as pd
import numpy as np
app = Flask(__name__)
app.secret_key = "hamza"
 


model_filename = "static\\model\\model.pkl"

loaded_model = joblib.load(model_filename)
print('Model Loaded Sucessfully')





def predict(model,audio_path):
    # Ignore all warnings
    warnings.filterwarnings("ignore")

    t_1 = feature_1(audio_path)
    t_2 = feature_2(audio_path)
    t_3 = feature_3(audio_path)

    # Concatenate the features horizontally
    extracted_features = pd.concat([t_1, t_2, t_3], axis=1, ignore_index=True)

    # Ensure the shape of the DataFrame is suitable for prediction
    extracted_features = extracted_features.values.reshape(1, -1)


    # Make predictions
    prediction = model.predict(extracted_features).item()

    return prediction







@app.route('/', methods=['GET','POST'])
def welcome_page():

    return render_template('index.html')



@app.route('/process_form', methods=['POST'])
def process_form():
    # Check if a file was included in the request
    if 'audio' not in request.files:
        return "No audio file provided in the request", 400

    uploaded_file = request.files['audio']

    # Make sure the file has a valid extension (you can specify the extensions you want to allow)
    allowed_extensions = {'mp3', 'wav', 'ogg'}  # Add audio file extensions here
    if '.' in uploaded_file.filename and uploaded_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
        # Save the file to a specific folder (e.g., 'static/uploads')
        uploaded_file.save(os.path.join('static/uploads', uploaded_file.filename))

        # Assuming your predict function takes the audio file path as an argument
        file_name = os.path.splitext(uploaded_file.filename)[0]
        audio_file_path = f'static/uploads/{uploaded_file.filename}'
        prediction=predict(loaded_model,audio_file_path)

        if prediction == 1:
            result = 'Abnormal'
        else:
            result = 'Normal'

        return render_template('result.html', prediction=result)
    else:
        return "Invalid audio file format. Supported formats: mp3, wav, ogg", 400



if __name__ == '__main__':
    app.run()


