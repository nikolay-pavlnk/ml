from flask import Flask, request, jsonify, json
from apartments_db import ApartmentsDB
from preprocessing import check_features
from predictor import Predictor
from utils.additional_models import LabelEncoder


web_app = Flask(__name__)
web_app.config["JSON_AS_ASCII"] = False
price_predictor = Predictor()
#db = ApartmentsDB()


@web_app.route("/api/v1/statistics")
def statistics():
    return jsonify(db.get_statistics())


@web_app.route("/api/v1/records", methods=['POST'])
def records():
    limit = request.args.get("limit", 10)
    offset = request.args.get("offset", 0)
    return jsonify(db.get_records(limit, offset))


@web_app.route("/api/v1/predict")
def predict():
    params = json.loads(request.args.get("json"))
    is_valid, features = check_features(params["features"])
    if is_valid:
        return price_predictor.predict(features, params["model"])
    else:
        return features


if __name__ == "__main__":
    web_app.run("0.0.0.0", 8080)
