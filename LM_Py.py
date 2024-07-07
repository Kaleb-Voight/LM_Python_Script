#!/usr/bin/env python
# coding: utf-8

# Linear Modeling Python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

print("\nPython Script Example\n")

input_file = sys.argv[1]

print(f"Reading {input_file}\n")

data = pd.read_csv(input_file)

data.head()

X = data[['x']]
Y = data[['y']]

print("Your Scatter Plot\n")
plt.scatter(X,Y, color = 'black')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Cool Data')
plt.show()
plt.savefig(f'Scatter of {input_file}.png')
plt.clf()
model = LinearRegression()

model.fit(X, Y)

y_pred = model.predict(X)

print("Your Regression Plot\n")
plt.scatter(X, Y, color='black')
plt.plot(X, y_pred, color='red')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Cool Data')
plt.show()
plt.savefig(f'Regression of {input_file}.png')