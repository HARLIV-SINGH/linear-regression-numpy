#Importing libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

#Loading data
df = pd.read_csv("housing.csv")
df_encoded = pd.get_dummies(df, columns=["ocean_proximity"]) #Encoding "ocean proximity" column to integers
df_filled = df_encoded.fillna(df_encoded.median()) #Filling NaN values with the column's median
df_shuffled = df_filled.sample(frac=1, random_state=100) #Shuffling the dataset to prevent bias

data = (df_shuffled.to_numpy()).astype(float) #Converting the dataframe to an array

rows = data.shape[0]

training_data = data[0:int((0.8*rows)),:] #Using 80% of the data for training

target = (training_data[:,8]).astype(float) #Extracting the target
y = target_scaled = target / 1e5 #Scaling the target

features = np.hstack((training_data[:, :8], training_data[:, 9:])) #Extracting the features
training_mean = features.mean(axis = 0)
training_std = features.std(axis = 0)
training_std[training_std == 0] = 1
x = features_scale = ((features - training_mean) / training_std) #Scaling the features

weights = (np.random.rand(features.shape[1]))*0.01 #Randomizing initial weights

bias = 0.0 #Initial bias = 0

learning_rate = 0.005 #learning rate
epochs = 2000 #number of loops

loss_history = [] #To track the loss function

for epoch in range(epochs):
    y_predicted = x @ weights + bias #Regression equation
    loss = np.mean((y_predicted - y)**2) #Loss function
    loss_history.append(loss) #Updating the loss function tracker

    grad_w = (2/x.shape[0]) * x.T @ (y_predicted - y) #Gradient wrt weights
    grad_b = (2/x.shape[0]) * np.sum(y_predicted - y) #Gradient wrt bias
    
    weights -= learning_rate*grad_w #Updating weights
    bias -= learning_rate*grad_b #Updating bias

    if epoch > 1 and abs(loss_history[-2] - loss_history[-1]) < 1e-5:
        print(f"Early stopping at epoch {epoch}")
        break

y_predicted_real = y_predicted * 1e5 #Rescaling

test_data = data[int(0.8*rows):,:].astype(float) #Using the rest of the data (20%) for testing

test_features = np.hstack((test_data[:, :8], test_data[:, 9:])).astype(float) #Extracting features
X = test_features_scales = ((test_features - training_mean) / training_std) #Scaling features

test_target = (test_data[:,8]).astype(float) #Extracting the target
y = test_target_scaled = test_target / 1e5 #Scaling the target

#final_weights = weights
#final_bias = bias

y_predicted_test = X@weights + bias #Regression quation
y_rescaled = y_predicted_test * 1e5 #Rescaling

rmse = np.sqrt(np.mean((y_rescaled - test_target)**2))
print("RMSE indicates the average magnitude of error between actual and predicted values")
print(f"RMSE: {rmse:.2f}\n")

mae = np.mean(np.abs(y_rescaled - test_target))
print("MAE calculates the average of the absolute differences between actual and predicted values")
print(f"MAE: {mae:.2f}\n")

ss_res = np.sum((test_target - y_rescaled) ** 2)
ss_tot = np.sum((test_target - np.mean(test_target)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print("R² score indicates how well the model fits the data")
print(f"R² Score: {r2:.4f}")
