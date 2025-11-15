import json
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from flask import Flask, request, jsonify
import psycopg2

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X,y)

def get_db_conn():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )

app = Flask(__name__)

@app.route("/")
def index():
    return "ML prediction service is running!"

@app.route("/predict")
def predict():
    try:
        x = float(request.args.get("x"))
    except:
        return jsonify({"error": "Please provide ?x=<number>"}), 400

    x_arr = np.array([[x]])
    pred = float(model.predict(x_arr)[0])

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (input_value, prediction) VALUES (%s, %s)",
        (x, pred)
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"input": x, "prediction": pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)