CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    input_value FLOAT,
    prediction FLOAT
);