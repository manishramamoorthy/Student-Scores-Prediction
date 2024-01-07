import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymongo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import argparse
import logging
import os
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

client = pymongo.MongoClient('mongodb://localhost:27017')
db = client["student_marks"]
collection = db['Student_scores']
documents = collection.find({})
all_data=[]
for i in documents:
   dict_data = {}
   dict_data['Hours'] = i['Hours']
   dict_data['Scores'] = i['Scores']
   all_data.append(dict_data)

df = pd.DataFrame(all_data)
print(df)

# creating the default paser arguments
def create_argument_parser():
    parser = argparse.ArgumentParser(description='parse arguments')

    parser.add_argument(
        '-H', '--host',
        type=str,
        default='0.0.0.0',
        help='hostname to listen on')

    parser.add_argument(
        '-P', '--port',
        type=int,
        default=5000,
        help='server port')
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='if given, enable or disable debug mode')

    return parser


# creating_app using the flask
def create_app(test_config=None):
    # create and configure the app
    flask_app = Flask(__name__)

    @flask_app.errorhandler(ValueError)
    def handle_value_error(ex):
        flask_app.logger.error(f"{ex}")
        response = jsonify({
            "message": f"{ex}"
        })
        response.status_code = 404
        return response

    @flask_app.errorhandler(KeyError)
    def handle_key_error(ex):
        flask_app.logger.error(f"{ex}")
        response = jsonify({
            "message": f"{ex}"
        })
        response.status_code = 404
        return response

    @flask_app.errorhandler(HTTPException)
    def handle_exception(ex):
        flask_app.logger.error(str(ex))
        response = jsonify({
            "message": ex.description
        })
        response.status_code = ex.code
        return response

    # todo: remove this handler
    @flask_app.errorhandler(Exception)
    def handle_exception(ex):
        flask_app.logger.exception(f"Unexpected runtime error: {ex}")
        response = jsonify({
            "message": "Unexpected runtime error"
        })
        response.status_code = 500
        return response

    # health check
    @flask_app.route('/', methods=['GET', 'POST'])
    def hello():
        return "Hello from gender prediction serving app"

    # predict score
    @flask_app.route(f'/student_scores_prediction/api/v1/all', methods=['POST'])
    def predict_score():
        try:
            req_json = request.get_json(force=True)
            hours = req_json['hours']  # Adjusted the key to match your input format
            hours_array = [[hours]]  # For compatibility with the model

            X = df[["Hours"]]
            y = df["Scores"]

            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

            LR = LinearRegression()
            LR.fit(X_train, y_train)

            prediction = LR.predict(hours_array)

            response = {"predicted scores": float(prediction[0])}
            return jsonify(response)

        except Exception as e:
            response = {"error": str(e)}
            return jsonify(response), 500
    return flask_app
if __name__ == '__main__':
    "run server"

    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    app = create_app()
    app.run(host=cmdline_args.host, port=cmdline_args.port, debug=cmdline_args.debug)