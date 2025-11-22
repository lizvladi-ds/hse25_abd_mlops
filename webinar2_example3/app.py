import os

import numpy as np
import psycopg2
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)


def get_db_conn():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


app = Flask(__name__)


@app.route("/")
def index():
    return "ML prediction service is running!"


@app.route("/predict")
def predict():
    x_val = request.args.get("x")

    if x_val is None:
        return jsonify({"error": "Please provide ?x=<number>"}), 400

    try:
        x = float(x_val)
    except ValueError:
        return jsonify({"error": "x must be a number"}), 400

    x_arr = np.array([[x]])
    try:
        pred = float(model.predict(x_arr)[0])
    except Exception as e:
        return jsonify({"error": f"Model failure: {str(e)}"}), 500

    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (input_value, prediction) VALUES (%s, %s)",
            (x, pred),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"DB error: {str(e)}"}), 500

    return jsonify({"input": x, "prediction": pred})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
