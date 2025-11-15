import json
from sklearn.linear_model import LinearRegression
import numpy as np
import os

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X,y)

x_test = np.array([[7]])
predict = model.predict(x_test)
print("Successful prediction")

output_dir = "output"
result = {
    "test_data": float(x_test[0][0]),
    "result": float(predict)
}

with open(os.path.join(output_dir, "prediction.txt"), "w") as f:
    f.write(json.dumps(result, indent=2))

print("Result saved successfully")
