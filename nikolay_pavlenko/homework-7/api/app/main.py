from flask import Flask, request, jsonify
from apartments_db import ApartmentsDB


web_app = Flask(__name__)
web_app.config["JSON_AS_ASCII"] = False
db = ApartmentsDB()


@web_app.route("/api/v1/statistics")
def statistics():
    return jsonify(db.get_statistics())


@web_app.route("/api/v1/records")
def records():
    limit = request.args.get("limit", 10)
    offset = request.args.get("offset", 0)
    return jsonify(db.get_records(limit, offset))


if __name__ == "__main__":
    web_app.run("0.0.0.0", 8080)
