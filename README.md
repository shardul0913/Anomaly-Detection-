Anomaly Detection
Approaches Considered: 
1.	Supervised learning- Using the data of preexisting anomalies or labeling the existing dataset XGboost or randomforest can be used in the supervised scenario. 
2.	Unsupervised: Predict the minority class using outlier detection techniques. The given unlabeled dataset it was important to learn from the ‘Normal’ machine data.
The current dataset is unlabeled, thus I chose to focus on the unsupervised classification for the anomaly detection. In order to create a robust implementation, I chose to ensemble the following algorithms.
Brief Overview
1.	Dimensionality reduction (PCA)
a.	By reducing the feature in the form of principal components with eigen / SVD decomposition we can trace the most amount of the variation for the datapoints in the dataset
b.	Thus, I chose to fit the PCA on the given data. Which is an unsupervised in nature. Thus by filtering the least contributing principal components from the PCA decomposition, we can filter out the outliers. 
c.	The PCA then were used to train linear and proximity based model

2.	Forecasting methods (SARIMA)
a.	Autoregressive Integrated Moving Average is the forecasting method where the autoregressive model(AR) uses the past forecasts to predict future values while moving average(MA) model does not uses the past forecasts to predict the future values whereas it uses the errors from the past forecasts.
Trend Parameters: 
p: Trend autoregression order.
d: Trend difference order.
q: Trend moving average order.
		Seasonal Parameters:
			     P: Seasonal autoregressive order.
     D: Seasonal difference order.
     Q: Seasonal moving average order.
      m: The number of time steps for a single seasonal period.
		The performance of the SARIMA can be improved by grid tuning the algorithm.		

3.	Neural Networks and Deep Learning Models (LSTM  Auto-encoder NN)
The final algorithms used in the ensemble solution is Long short-term memory (LSTM) is an artificial recurrent neural network with autoencoding approach. 
The data at hand is the time series thus instead of choosing standard architecture I decided to use LSTM for its decision making based on the the current input, previous output and previous memory. The type of LSTM used was multivariate. 
The input sensor vectors were normalized between 0 and 1 and then passed to the LSTM in 3 dimensional format just as it requires [data samples, time steps, features]. Each time LSTM takes the all 4 sensor data at a time.
The architecture of the Neural network was implanted by implementing the RepeatVector.
The activation function used is ReLU to weight the predictions as it is to the next Neuron. 

4.	Linear Models for Outlier Detection (vOne-Class)
a.	How much the data deviates from the true boundary 
b.	By using the one class SVM we can trace the high dimensional boundary plane a circle in this case to classify the data 

5.	Proximity-Based Outlier Detection Models (Kmeans)
a.	Based on the interaction in the higher dimensional plane. Cosine similarity / Euclidean similarity, how much deviation does a outlier have
b.	As the proximity based algorithms are sensitive to outliers, this property can be used to track the anomaly in the data points
c.	It does not completely classify the inter class relation between the 4 features.
d.	Hyperparameter tuning of number of clusters is dataset specific.
