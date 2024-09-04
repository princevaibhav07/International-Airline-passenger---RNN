# International Airline Passenger Prediction using RNN

## Overview

This project focuses on time series forecasting using a Recurrent Neural Network (RNN) to predict the number of international airline passengers. The project is implemented in Python using Jupyter Notebook and leverages libraries such as TensorFlow and Keras for model building.
Introduction
Time series forecasting is a crucial task in many fields, and predicting airline passenger numbers can be essential for resource planning and management. In this project, we build an RNN model to forecast the number of international airline passengers based on historical data.

## Dataset

The dataset used in this project contains monthly totals of international airline passengers between 1949 and 1960. The data is univariate and non-stationary, making it a good candidate for RNN-based forecasting.

Source: The dataset is available as part of the AirPassengers dataset in R and is also widely used in machine learning tutorials.
Data Preprocessing
Before feeding the data into the RNN model, the following preprocessing steps were performed:

Normalization: The data was normalized to ensure that the values fall within a specific range, which improves the performance of the model.
Train-Test Split: The data was split into training and testing sets, ensuring that the model is trained on one portion of the data and tested on another.
Model Architecture
The RNN model used in this project consists of the following layers:

Input Layer: Accepts the time series data.
LSTM Layers: Two stacked LSTM (Long Short-Term Memory) layers, which are well-suited for time series data due to their ability to learn long-term dependencies.
Dense Layer: A fully connected layer to produce the final output.
Output Layer: Predicts the number of passengers for the next time step.

## Training the Model

The model was trained using the Adam optimizer and Mean Squared Error (MSE) as the loss function. The training process included:

Batch Size: 32
Epochs: 50
Validation Split: 20% of the training data was used for validation.

## Evaluation

The model was evaluated using the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics. The evaluation results demonstrate the effectiveness of the model in predicting future passenger numbers.

## Results

The model successfully captures the trend and seasonal patterns in the data. The predictions align closely with the actual passenger numbers, demonstrating the potential of RNNs in time series forecasting.
