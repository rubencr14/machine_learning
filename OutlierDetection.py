"""
This module will have all the methods for detecting outliers
"""

import numpy as np
import pandas as pd

try:
	from sklearn.ensemble import IsolationForest
	from sklearn.covariance import EllipticEnvelope
	from sklearn.neighbors import LocalOutlierFactor

except ImportError:
	print("sklearn library not found, download it via pip or conda!")

	

__author__ = "ruben canadas rodriguez"



class InterQuantileRangeOutliersDetection:

	"""This class uses standard deviation and interquantile ranges for finding outliers in the dataset"""

	def __init__(self, data, variable):
		
		self.__outlier_variable = variable
		self.__return_bounds = True
		self.__data = data
		self.__nstd = 2 # Number of standard deviations to consider an outlier
		self.__data_mean = self.__data[self.__outlier_variable].mean()
		self.__k = 1.0 # Cutoff for interquantile
		self.__data_std = self.__data[self.__outlier_variable].std()
		self.__cutoff = self.__data_std * self.__nstd
		self.__q25 = None
		self.__q75 = None




	def OutlierDetectionStandard(self):

		"""
		This method uses standard deviation to find outliers ( the next method will use interquantil ranges)
		"""

		lower, upper = self.__data_mean - self.__cutoff, self.__data_mean + self.__cutoff
		return lower, upper, self.__data.loc[(self.__data[self.__outlier_variable] > upper) | (self.__data[self.__outlier_variable] < lower)] #Return the values from dataframe that are greater than upper bound and lower than lower bound




	def OutlierDetectionInterquantile(self):

		"""
		This method uses interquantil ranges to find outliers
		"""

		self.__q25, self.__q75 = self.__data[self.__data[self.__outlier_variable] < self.__data[self.__outlier_variable].quantile(0.25)], self.__data[self.__data[self.__outlier_variable] < self.__data[self.__outlier_variable].quantile(0.75)]
		self.__q25, self.__q75 = max(self.__q75[self.__outlier_variable]), max(self.__q25[self.__outlier_variable])
		cutoff = (self.__q75-self.__q25)*self.__k
		upper, lower = self.__q25 - cutoff, self.__q75 + cutoff
		return lower, upper, self.__data.loc[(self.__data[self.__outlier_variable] > upper) | (self.__data[self.__outlier_variable] < lower)]




class IsolationForestScore:


	def __init__(self,df,variable):

		self.__df = df
		self.__variable = variable

	def IsolationForest(self):

		isolation_forest = IsolationForest()
		isolation_forest.fit(self.__df[self.__variable].values.reshape(-1,1))
		xx = np.linspace(self.__df[self.__variable].min(), self.__df[self.__variable].max(), len(self.__df)).reshape(-1,1)
		anomaly_score = isolation_forest.decision_function(xx)
		outlier = isolation_forest.predict(xx)


		return xx, anomaly_score, outlier



class EllipticEnvelopeOutlierDection:

	"""
	Using this method we assume attributes are Gaussian distributed, we can transform them before, using
	the Gaussian transformer in FeatureEngineering module, and then applies the elliptic envelope method.
	"""


	def __init__(self, df, x, y,drop_outliers=False):

		self.__x = x
		self.__y = y
		self.__df = df
		self.__factor = 2 #Distance betweem the last value and the maximum to plot 
		self.__drop_outliers = drop_outliers
		


	def DetectOutliersUsingEnvelope(self):

		data = self.__df[[self.__x, self.__y]].values
		clf = EllipticEnvelope()
		x_min_value, x_max_value = min(self.__df[self.__x].values) - self.__factor, max(self.__df[self.__x].values) + self.__factor
		y_min_value, y_max_value = min(self.__df[self.__y].values) - self.__factor, max(self.__df[self.__y].values) + self.__factor
	
		xx, yy = np.meshgrid(np.linspace(x_min_value, x_max_value, 500), np.linspace(y_min_value, y_max_value, 500))
		clf.fit(data)
		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		pred = clf.fit_predict(data) #Outliers = -1, inliers = 1
		if self.__drop_outliers: # Let's drop outliers from dataset!
			for index, outlier in enumerate(pred):
				if outlier == -1:
					self.__df = self.__df.drop(index, axis=0)
			return self.__df 

		else:
			return xx, yy, Z



class MultiVariateOutlierDetection:

	"""This class applies local outlier factor technique to find the outliers"""

	def __init__(self, df):
		
		self.__df = df


	def LocalOutlier(self):

		outlier_model = LocalOutlierFactor(n_neighbors=2, metric="euclidean")
		y_pred = outlier_model.fit_predict(self.__df)
		X_scores = outlier_model.negative_outlier_factor_
		return X_scores
