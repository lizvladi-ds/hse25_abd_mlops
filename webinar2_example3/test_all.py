import os
import psycopg2
import pytest
from app import app, model


# ------------------------------------------------------
# FIXTURES
# ------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ------------------------------------------------------
# UNIT TESTS — API CONTRACT
# ------------------------------------------------------

def test_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"ML prediction service is running!" in resp.data


def test_predict_contract(client, mocker):
    mock_conn = mocker.Mock()
    mock_cursor = mocker.Mock()
    mock_conn.cursor.return_value = mock_cursor
    mocker.patch("app.get_db_conn", return_value=mock_conn)

    resp = client.get("/predict?x=3")
    assert resp.status_code == 200

    data = resp.get_json()
    assert "input" in data
    assert "prediction" in data
    assert isinstance(data["input"], float)
    assert isinstance(data["prediction"], float)


# ------------------------------------------------------
# UNIT TESTS — INPUT VALIDATION
# ------------------------------------------------------

@pytest.mark.parametrize("value", ["abc", None, {}, []])
def test_predict_invalid_input(client, value):
    resp = client.get(f"/predict?x={value}")
    assert resp.status_code == 400


def test_predict_missing_input(client):
    resp = client.get("/predict")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


# ------------------------------------------------------
# UNIT TESTS — MODEL ERRORS
# ------------------------------------------------------

def test_model_failure(client, mocker):
    mock_model = mocker.Mock()
    mock_model.predict.side_effect = Exception("Model error")
    mocker.patch("app.model", mock_model)

    resp = client.get("/predict?x=1")
    assert resp.status_code in [400, 500]


# ------------------------------------------------------
# UNIT TESTS — SQL CONTRACT
# ------------------------------------------------------

def test_sql_insert_contract(client, mocker):
    mock_conn = mocker.Mock()
    mock_cursor = mocker.Mock()
    mock_conn.cursor.return_value = mock_cursor

    mocker.patch("app.get_db_conn", return_value=mock_conn)

    client.get("/predict?x=10")

    mock_cursor.execute.assert_called_once()

    query, params = mock_cursor.execute.call_args[0]
    assert "INSERT INTO predictions" in query
    assert isinstance(params[0], float)
    assert isinstance(params[1], float)


# ------------------------------------------------------
# UNIT TESTS — DB FAILURE
# ------------------------------------------------------

def test_db_unavailable(client, mocker):
    mocker.patch("app.get_db_conn", side_effect=Exception("DB down"))

    resp = client.get("/predict?x=5")
    assert resp.status_code in [400, 500]
    assert isinstance(resp.get_json(), dict)


# ------------------------------------------------------
# UNIT TESTS — MODEL VALUE TEST
# ------------------------------------------------------

def test_model_prediction_linear():
    pred = model.predict([[4]])[0]
    assert pred == 8  # модель обучена на y = 2x


# ------------------------------------------------------
# INTEGRATION TEST — REQUIRES REAL POSTGRES
# ------------------------------------------------------
# Run with: pytest -m integration

@pytest.mark.integration
def test_predict_inserts_row(client):
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM predictions")
    before = cur.fetchone()[0]

    resp = client.get("/predict?x=5")
    assert resp.status_code == 200

    cur.execute("SELECT COUNT(*) FROM predictions")
    after = cur.fetchone()[0]

    assert after == before + 1

    cur.close()
    conn.close()
