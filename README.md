# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe 
2.Write a function computeCost to generate the cost function. 
3.Perform iterations og gradient steps with learning rate. 
4.Plot the cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Parani Bala M
RegisterNumber:  212224230192
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1,1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = (X).dot(theta).reshape(-1, 1)

        # Calculate errors
        errors = (predictions - y).reshape(-1,1)

        # Update theta using gradient descent
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta


# Load dataset
data = pd.read_csv('50_Startups.csv', header=None)
print(data.head())

# Assuming the last column is your target variable 'y' and the preceding columns are your features 'X'
X = (data.iloc[1:, :-2].values)
print(X)

X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1,1)
print(y)

X1_Scaled = scaler.fit_transform(X1)
y1_Scaled = scaler.fit_transform(y)
print('Register No.:')
print(X1_Scaled)

# Learn model parameters
theta = linear_regression(X1_Scaled, y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
<img width="759" height="239" alt="Screenshot 2025-09-01 151651" src="https://github.com/user-attachments/assets/ae07ba6e-3b0c-4711-8c41-9e2142e99562" />
<img width="770" height="397" alt="Screenshot 2025-09-01 151632" src="https://github.com/user-attachments/assets/7a60cb85-1e1e-47db-9d62-e63e35528010" />
<img width="902" height="400" alt="Screenshot 2025-09-01 151623" src="https://github.com/user-attachments/assets/95abd025-b6b5-4f15-a15d-37ac7218ec6a" />
<img width="805" height="404" alt="Screenshot 2025-09-01 151612" src="https://github.com/user-attachments/assets/23b9c443-5090-42f1-a743-b31dc53c26af" />
<img width="611" height="406" alt="Screenshot 2025-09-01 151600" src="https://github.com/user-attachments/assets/9a8bb552-e9ff-495a-8b38-3a7cb71ea17e" />
<img width="706" height="394" alt="Screenshot 2025-09-01 151543" src="https://github.com/user-attachments/assets/bef86b38-d62b-4bcd-aeda-fdcbca664f13" />
<img width="653" height="382" alt="Screenshot 2025-09-01 151533" src="https://github.com/user-attachments/assets/77eeda22-9955-41ef-90a1-3060641c0d89" />
<img width="650" height="406" alt="Screenshot 2025-09-01 151515" src="https://github.com/user-attachments/assets/2abb5733-f91b-47f6-8e18-895934bf56f9" />
<img width="845" height="412" alt="Screenshot 2025-09-01 151503" src="https://github.com/user-attachments/assets/52f066d7-eb6b-473e-84aa-113bb61c2c50" />
<img width="630" height="181" alt="Screenshot 2025-09-01 145856" src="https://github.com/user-attachments/assets/6c5e5bd0-5142-49a2-b8bb-dacc2afa1b4e" />
<img width="950" height="289" alt="Screenshot 2025-09-01 145846" src="https://github.com/user-attachments/assets/07659f61-87f4-4530-8b9f-006c9322ebcb" />
<img width="908" height="313" alt="Screenshot 2025-09-01 145837" src="https://github.com/user-attachments/assets/734e26ed-0054-4b7c-b10a-cce6a8be0192" />
<img width="848" height="304" alt="Screenshot 2025-09-01 145827" src="https://github.com/user-attachments/assets/378830f4-2878-418b-9ade-52243a346951" />
<img width="628" height="307" alt="Screenshot 2025-09-01 145815" src="https://github.com/user-attachments/assets/707338ca-413a-4c0a-bccf-9485983fa389" />
<img width="657" height="304" alt="Screenshot 2025-09-01 145807" src="https://github.com/user-attachments/assets/967f342f-4b87-4847-9277-9f5478ea82e0" />
<img width="1006" height="289" alt="Screenshot 2025-09-01 145759" src="https://github.com/user-attachments/assets/d7be935a-643a-41da-8173-118c4777e342" />
<img width="1032" height="303" alt="Screenshot 2025-09-01 145750" src="https://github.com/user-attachments/assets/688cb762-6fb3-4981-be7a-d530b6ed27d9" />
<img width="1024" height="307" alt="Screenshot 2025-09-01 145743" src="https://github.com/user-attachments/assets/363c1bcb-1c45-4429-9491-1368c2921565" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
