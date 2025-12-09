import os
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI

# 1. Указываем, ГДЕ находится MLflow Tracking Server
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI","http://mlflow-service:5000")
os.environ["AWS_ENDPOINT_URL"] = os.getenv("AWS_ENDPOINT_URL","https://storage.yandexcloud.net")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "ru-central1")

# 2. Имя и стадия модели
MODEL_NAME = "webinar_5"
MODEL_STAGE = "Production"

# 3. Загружаем модель из Model Registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri)

# 4. FastAPI-сервис
app = FastAPI()

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame(payload["inputs"])
    df = df.astype(float)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
