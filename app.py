from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import json

app = Flask(__name__)
CORS(app)

# Load the tokenizer and model
token_form = pickle.load(open('tokenizer.pkl', 'rb'))
model = tf.keras.models.load_model('model.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['post_content']
    twt = [sentence]
    twt = token_form.texts_to_sequences(twt)
    twt = pad_sequences(twt, maxlen=50)

    # Predict the post content
    prediction = model.predict(twt)[0][0]

    # Decide the result
    result = "Potential Suicide Post" if prediction > 0.5 else "Non Suicide Post"

    # Prepare data for chart
    class_label = ["Potential Suicide Post", "Non Suicide Post"]
    prob_list = [prediction * 100, 100 - prediction * 100]
    prob_dict = {"label": class_label, "probability": prob_list}
    df_prob = pd.DataFrame(prob_dict)

    # Create bar chart
    fig = px.bar(df_prob, x='label', y='probability')
    fig.update_layout(title_text="Our Model Prediction Comparison")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Return result and chart
    return jsonify(result=result, chart=graphJSON)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
#
