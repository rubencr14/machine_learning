import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, skew
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVC, SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, RobustScaler, \
									StandardScaler, PolynomialFeatures, KBinsDiscretizer, Binarizer, FunctionTransformer
try:
	from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
	print("pathos needs to be downloaded for multiprocessing")
from sklearn.decomposition import PCA
try:
	from tpot import TPOTClassifier, TPOTRegressor
except ImportError:
	print("AutoML cannot be used since tpot is not installed")
import numpy as np
import matplotlib.pyplot as plt
import Configuration as C
import pandas as pd
import pickle, os
import warnings
import multiprocessing as mp
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, SGDClassifier
from warnings import simplefilter
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


__author__="Ruben Canadas"
__mail__="rubencr14@gmail.com"
__maintainer__="Ruben Canadas"
__version__=1.0


"""
This module contains all the methods necessary for carrying out feature engineering tasks. It includes several methods 
for feature selection (RFE, sequential...), methods for power transforming (log, box-cox, Gaussian..), for scaling and 
(minmax, standardize, robust...) and others such as dimensionality reduction via PCA or polynomial interactions

"""




class Error(Exception):

	"""Base class for other error exceptions"""
	pass

class MethodNotFoundError(Error):

	pass

########################################################################################################################
#                           	FEATURE SELECTION!
########################################################################################################################


class FeatureSelectionRFECV(BaseEstimator, TransformerMixin):

	"""This method perform recursive feature elimination selecting the best choice of number of feature
	using cross-validation"""

	def __init__(self,  pipe, procs=2):

		self.__method = C.TYPE
		if not self.__method in ["reg", "clf"]:
			raise ValueError("Method {} not available, pick reg or clf".format(self.__method))
		if self.__method == "clf":
			self.__estimator = SGDClassifier()
			self.__cv = StratifiedKFold(n_splits=10)
			self.__scoring = ["accuracy"]
		elif self.__method == "reg":
			self.__estimator = SGDRegressor()
			self.__cv = 10
			self.__scoring = ["r2"]
		self.__num_procs = procs
		self.__y = None
		self.__target = C.TARGET
		self.__pipe = pipe

	def _save_columns_to_drop(self, columns_to_drop=None, action="write"):
		if action=="write":
			outfile = open(os.path.join(C.PATH, "columns_to_drop_{}.pkl".format(self.__pipe)), "wb")
			pickle.dump(columns_to_drop, outfile); outfile.close()
		elif action=="read":
			infile = open(os.path.join(C.PATH, "columns_to_drop_{}.pkl".format(self.__pipe)), "rb")
			to_drop = pickle.load(infile); infile.close()
			return to_drop

	def fit(self, X, y=None):
		infile = open(os.path.join(C.PATH, "train.pkl"), "rb")
		train = pickle.load(infile); infile.close()
		self.__y = train[self.__target]
		return self

	def transform(self,X):


		if not os.path.exists(os.path.join(C.PATH, "columns_to_drop_{}.pkl".format(self.__pipe))):

			rfecv = RFECV(self.__estimator, step=1, min_features_to_select=10, cv=self.__cv,  n_jobs=self.__num_procs)
			rfecv.fit(X,self.__y)
			ranking = rfecv.ranking_
			X_new = X.copy()
			columns_to_drop = []
			for rank, feat in zip(ranking, X.columns.values):
				if rank != 1:
					columns_to_drop.append(feat)
					X_new = X_new.drop([feat], axis=1)
			self._save_columns_to_drop(columns_to_drop=columns_to_drop, action="write")
			return X_new
		else:
			print("applying existing RFE")
			X_new = X.copy()
			to_drop = self._save_columns_to_drop(action="read")
			X_new = X_new.drop(to_drop, axis=1)
			return X_new



#CUSTOM CLASS CREATED TO SELECT THE BEST NUMBER OF FEATURES
class FeatureSelectionRFE(BaseEstimator, TransformerMixin):

	"""This class selects best features according to recursive feature selection using paralelization and
	custom CV"""

	def __init__(self, method="reg", verbose=False, procs=10, plot=True):

		self.__method = method
		if not self.__method in ["reg","clf"]:
			raise ValueError("Method {} not in reg, clf".format(self.__method))
		if self.__method == "clf":
			self.__estimator = SVC(kernel="linear")
			self.__cv = StratifiedKFold(n_splits=10)
			self.__scoring = ["accuracy"]
		elif self.__method == "reg":
			self.__estimator = SGDRegressor()
			self.__cv = 10
			self.__scoring = ["r2"]
		self.__verbose = verbose
		self.__num_procs = procs
		self.__plot = plot
		self.__X = None
		self.__y = None
		self.__X_new = None
		self.__target = C.TARGET

	def __str__(self):

		"""Class for defining the characteristics of the object"""

		return "FEATURE SELECTION CLASS OBJECT  Method: {}, Verbosity: {}, Processors: {}, Plot: {}".format(self.__method,\
			self.__verbose, self.__num_procs, self.__plot)

	@staticmethod
	def plot(values):
		nums = [val[0] for val in values]
		scores = [val[1] for val in values]
		plt.plot(nums,scores);plt.show()

	@staticmethod
	def get_best_score(scores):

		"""This method return the columns to drop given a dictionary with {"num":[cross_val_score, to_drop_columns]}"""
		try:
			maximum = max([val[0] for val in scores.values()])
			for key,value in scores.items():
				if value[0] == maximum:
					return value[1] #Columns to drop
		except:
			pass

	#FIXME: DEFINE X VARIABLE!!
	def compute_rfe(self, num):

		"""This method, given number of features num, performs a 
		Recursive Feature Elimination (RFE) and returns a list:
		[num_of_features, mean of cross val scores, columns to drop]"""

		try:
			infile = open(os.path.join(C.PATH, "train.pkl"), "rb")
			df = pickle.load(infile); infile.close()
			self.__y = df[self.__target]
			if self.__verbose: print("NUMBER OF FEATURES: ", num)
	
			selector = RFE(self.__estimator, num, step=1)
			selector = selector.fit(self.__X, self.__y)
			ranking = selector.ranking_
			columns = self.__X.columns.values
			to_drop = []

			for rank,feat in zip(ranking, columns):
				if rank != 1:
					to_drop.append(feat)
			self.__X_new = self.__X.drop(to_drop, axis=1)
			scores_val = cross_validate(self.__estimator, self.__X_new, self.__y, cv=self.__cv,
										scoring=self.__scoring)

			if self.__verbose: print("Score using {} features is {}".format(num, scores_val["test_r2"].mean()))
			return [num, scores_val["test_r2"].mean(), to_drop]
		except Exception as e:
			print("Not working for num: {} with error {}".format(num, e))
			return [None, None, None] #If there's an error return a None array


	def fit(self,X,y=None):
		self.__X = X
		return self

	def transform(self,X,y=None):

		number_of_features = self.__X.shape[1]
		list_features = np.arange(5, number_of_features, 1)
		pool = mp.Pool(self.__num_procs)
		results = pool.map(self.compute_rfe, list_features)
		scores = {"{}".format(value[0]):[value[1], value[2]] for value in results if value[0] is not None}
		best_feature_drop = FeatureSelectionRFE.get_best_score(scores)
		final_X = self.__X.drop(best_feature_drop, axis=1)
		infile = open(os.path.join(C.PATH, "rfe_drop.pkl"), "wb")
		pickle.dump(best_feature_drop, infile); infile.close()
		return final_X


class FeatureSelectionTreeBased(BaseEstimator, TransformerMixin):

	"""Embedded feature selection using random forest for estimating feature importances"""

	def __init__(self, method="clf", verbose=True, num_procs=25, plot=True):

		self.__method = method
		if not self.__method in ["reg","clf"]:
			raise ValueError("Method {} not in reg, clf".format(self.__method))
		if self.__method == "clf": self.__estimator = RandomForestClassifier()
		elif self.__method == "reg": self.__estimator = RandomForestRegressor()
		self.__verbose = verbose
		self.__num_procs = num_procs
		self.__plot = plot

	def fit(self,X,y=None):
		return self

	def transform(self,X,y=None):
		
		self.__estimator.fit(X,y)
		best_score  = max(self.__estimator.feature_importances_)
		for feat, score in zip(X.columns.values, self.__estimator.feature_importances_):
			if score < 0.30 * best_score:
				X = X.drop([feat],axis=1)
		return X


class FeatureSelectionSequentianElimination(BaseEstimator, TransformerMixin):

	"""This class selects best features using wrapper method called sequential elimination"""

	def __init__(self, pipe="pipe_1", types="SFS"):
		print(" Feature selection (Seaquential elimination)...")
		self.__kernel = "rbf"
		self.__nfeatures = 20
		self.__forward_floating = types
		self.__scoring = "accuracy"
		self.__cv = 10
		self.__pipe = pipe
		self.__y = None

	def fit(self, X,y=None):

		if len(X.columns.values) > 40:
			self.__nfeatures = 30
		else:
			self.__nfeatures = 20
		self.__y = y

		return self

	def transform(self, X, y=None):

		if not os.path.exists(os.path.join(C.PATH, "columns_{}_{}.pkl".format(self.__pipe, self.__forward_floating))):

			if self.__forward_floating == "SFS":
				forward = True
				floating = False 

			elif self.__forward_floating == "SBS":
				forward = False
				floating = False

			elif self.__forward_floating == "SFFS":
				forward = True
				floating = True

			elif self.__forward_floating == "SBFS":
				forward = False
				floating = True

			sfs = SFS(SVC(kernel = self.__kernel), 
					k_features=self.__nfeatures, 
					forward=forward, 
					floating=floating, 
					verbose=0,
					scoring=self.__scoring,
					cv=self.__cv, n_jobs=20)

			infile = open(os.path.join(C.PATH ,"train.pkl"), "rb")
			train = pickle.load(infile); infile.close()
			self.__y = train[C.TARGET]
			sfs.fit(X,self.__y)
			best = list(sfs.k_feature_names_)

			# The features that are not important according 
			# to the algorithm are discared in the outputted csv file.
			X_final = X.copy()
			for elem in X.columns.values:
				if elem not in best:
					X_final = X_final.drop([elem], axis=1)

			outfile = open(os.path.join(C.PATH, "columns_{}_{}.pkl".format(self.__pipe, self.__forward_floating)), "wb")
			pickle.dump(X_final.columns.values, outfile); outfile.close()
			#print("new {}   length: {}".format(new_columns, len(new_columns)))
			return X_final
			#return X_final
		else:
			X_new = X.copy()
			infile = open(os.path.join(C.PATH, "columns_{}_{}.pkl".format(self.__pipe, self.__forward_floating)), "rb")
			to_drop = pickle.load(infile); infile.close()
			X_new = X_new.drop(to_drop, axis=1)
			return X_new


########################################################################################################################
#                           	POWER TRANSFORMATIONS:
########################################################################################################################


class PowerTransformationsWrapper(BaseEstimator, TransformerMixin):

	"""This class applies power transformations and analyzes if the feature is Gaussian distributed or skewed using
	statistics tests"""

	def __init__(self, method):
		self.__method = method
		self.__feature = None

	def __str__(self):

		return "TRANSFORMER OBJECT {}".format(self.__method)

	@property
	def available_methods(self):

		return "{}".format(["BoxCox", "YeoJohnson", "Log", "Gaussian"])


	def is_normal_distributed(self, data_array):
		"""Saphiro-Wilk test is used to know if a feature is Gaussian distributed"""
		
		alpha = 0.05
		stat, p_value = shapiro(data_array) #Data has to be array-shaped
		if p_value > alpha:
			#print("Feature looks Gaussian-distributed")
			return True
		else:
			#print("Feature does not look Gaussian-distributed")
			return False


	def is_right_skewed(self, data_array):

		""" Returns True if the distribution is rigth-skewed, returns False if
		it is left-skewed, returns None if it is normally distributed"""

		if skew(data_array) > 0:
			return True
		elif skew(data_array) < 0:
			return False
		else:
			return None	

	def fit(self,X,y=None):
		return self


	def transform(self,X):

		self.__feature = X.columns.values
		X_new = X.copy()
		for feat in self.__feature:
			if self.is_right_skewed(X_new[feat].values):
				if self.__method == "BoxCox":
					X_new[feat] = BoxCoxTransform(feature=feat).fit_transform(X_new)
				elif self.__method == "YeoJohnson":
					X_new[feat] = YeoJohnsonTransform(feature=feat).fit_transform(X_new)
				elif self.__method == "Log":
					X_new[feat] = LogTransform(feature=feat).fit_transform(X_new)
			if not self.is_normal_distributed(X_new[feat].values):
				X_new[feat] = GaussianTransformer(feature=feat).fit_transform(X_new)
			else:
				pass
		return X_new


class BoxCoxTransform(BaseEstimator, TransformerMixin):

	"""Power transforms minimizes the skewness. Box-Cox only allows positive values. If lambda parameters is None
	the algorithm finds the lambda that maximizes  the log-likelihood function"""

	def __init__(self, feature=None):
		
		self.__feature = feature
		if self.__feature is None:
			exit("Please select the feature to transform")

	@staticmethod
	def __check_plot(X,X_cox,feature):

		fig, (ax1, ax2) = plt.subplots(2)
		sns.distplot(X[feature].values, ax=ax1)
		sns.distplot(X_cox, ax=ax2)
		plt.show()
		

	def fit(self, X,y=None):
		return self

	def transform(self, X):

		try:
			X_new = X.copy()
			boxcox, lambda_param = stats.boxcox(X_new[self.__feature].values)
			X_new[self.__feature] = boxcox
			return boxcox
		except:
			#print("Negative data not allowed")
			return X[self.__feature].values


class YeoJohnsonTransform(BaseEstimator, TransformerMixin):

	"""YeoJohnson is a type of power transformation"""

	def __init__(self, feature=None):
		
		self.__feature = feature
		if self.__feature is None:
			exit("Please select the feature to transform")

	@staticmethod
	def __check_plot(X,X_cox,feature):

		fig, (ax1, ax2) = plt.subplots(2)
		sns.distplot(X[feature].values, ax=ax1)
		sns.distplot(X_cox, ax=ax2)
		plt.show()
		

	def fit(self, X,y=None):
		return self

	def transform(self, X):
		try:
			yeojohnson, lambda_param = stats.yeojohnson(X[self.__feature])
			return yeojohnson
		except:
			return X[self.__feature].values

class LogTransform(BaseEstimator, TransformerMixin):

	"""LogTransform converts values in logarithmic scale. Using Cox-Box with lambda equals to zero
	we obtain the logarithmic scale"""

	def __init__(self, feature=None):
		
		self.__feature = feature
		self.__lambda = 0
		if self.__feature is None:
			exit("Please select the feature to transform")

	@staticmethod
	def __check_plot(X,X_cox,feature):

		fig, (ax1, ax2) = plt.subplots(2)
		sns.distplot(X[feature].values, ax=ax1)
		sns.distplot(X_cox, ax=ax2)
		plt.show()
		

	def fit(self, X,y=None):
		return self

	def transform(self, X):

		min_value = X[self.__feature].min()
		try:
			log = stats.boxcox(X[self.__feature], lmbda=self.__lambda)
			return log
		except:
			#print("Negative data not allowed")
			return X[self.__feature].values

class GaussianTransformer(BaseEstimator, TransformerMixin):

	"""This class transform a non-Gaussian distributed feature to a Gaussian distributed one"""

	def __init__(self, feature=None):
		self.__feature = feature

	@staticmethod
	def __check_plot(X,X_cox,feature):

		fig, (ax1, ax2) = plt.subplots(2)
		sns.distplot(X[feature].values, ax=ax1)
		sns.distplot(X_cox, ax=ax2)
		plt.show()

	def fit(self,X,y=None):
		return self

	#TODO: FINISH THE METHOD TRANSFORM!
	def transform(self, X):
		qt = QuantileTransformer(output_distribution="normal")
		quantile_feat = qt.fit_transform(X[self.__feature].values.reshape(-1,1))
		X[self.__feature] = quantile_feat
		return quantile_feat



########################################################################################################################
#                           	FEATURE SCALING
########################################################################################################################

class FeatureScaling(BaseEstimator, TransformerMixin):

	"""This classes perfroms feature scaling which are almost always useful for improving predictions. Except for
	decision trees-based methods (ADA, random forest, xgboost...)"""

	def __init__(self, scaler_type="none", idx_scaler=None):
		self.__scaler_type = scaler_type
		self.__available_scalers = ["std", "robust", "min_max", "none"]
		if not self.__scaler_type in self.__available_scalers:
			raise MethodNotFoundError("Method {} is not available, pick one of these: {}".format(self.__scaler_type,
																								 self.__available_scalers))
		self.__idx_scaler = idx_scaler
		self.__X_final = None

	@staticmethod
	def __scaler_pickler(scaler_object, name):

		outfile = open(os.path.join(C.PATH, name), "wb")
		pickle.dump(scaler_object, outfile)
		outfile.close()

	def fit(self,X,y=None):
		return self

	def transform(self,X, y=None):

		if X is None: 
			exit("X was none...exiting..")
		columns = X.columns.values
		if self.__scaler_type == "std":
			scaler = StandardScaler()
			scaler.fit(X)
			FeatureScaling.__scaler_pickler(scaler,"std_scaler_{}.pkl".format(self.__idx_scaler))
			self.__X_final = scaler.transform(X)

		elif self.__scaler_type == "robust":
			scaler = RobustScaler()
			self.__X_final = scaler.fit_transform(X)
			FeatureScaling.__scaler_pickler(scaler, "robust_scaler_{}.pkl".format(self.__idx_scaler))

		elif self.__scaler_type == "min_max":
			scaler = MinMaxScaler()
			self.__X_final = scaler.fit_transform(X)
			FeatureScaling.__scaler_pickler(scaler, "minmax_scaler_{}.pkl".format(self.__idx_scaler))

		elif self.__scaler_type == "none":
			self.__X_final = X

		try:
			return pd.DataFrame(self.__X_final, columns=columns)
		except Exception as e:
			print("exception is ", e)
		



class CreatePolynomials(BaseEstimator, TransformerMixin):

	"""This class creates new feature combining the original ones using polynomials"""

	def __init__(self):

		self.__degree = 2

	def fit(self,X,y=None):
		return self

	def transform(self,X):

		poly = PolynomialFeatures(degree=self.__degree, interaction_only=True)
		X_new = poly.fit_transform(X)
		columns = ["feat_{}".format(i) for i in range(X_new.shape[1])]
		try:
			return pd.DataFrame(X_new, columns=columns)
		except Exception as e:
			pritn("exception is ", e)


#TODO: DEFINE THE FEATURE!
class BinningTransformer(BaseEstimator, TransformerMixin):

	"""Like categorical data can be encoded into numerical, numerical features can be decoded
	into categorical features. The two main methods are named binarization and discretization.
	Discretization also known as binning, divides the continous feature into a pre-specified number of
	categories (bins)"""

	def __init__(self, num_bins=5, encoder="onehot", strategy="quantile"):
		self.__num_bins = num_bins
		self.__encoder = encoder
		self.__strategy = strategy

	def __str__(self):

		"""Information of the Binning object"""

		return "BINNING TRANSFORMER OBJECT: bins: {}, encoder: {}, strategy: {}".format(self.__num_bins, self.__encoder, self.__strategy)


	def fit(self,X,y=None):
		return  self

	def transform(self,X):

		discretizer = KBinsDiscretizer(n_bins=self.__num_bins, encode=self.__encoder, strategy=self.__strategy)
		return discretizer.fit_transform(X)



class BinarizerTransform(BaseEstimator, TransformerMixin):

	"""All values below or equal to the threshold are replaced by 0, above it by 1.
	This is called Binarization"""

	def __init__(self,threshold,feature=None):

		self.__threshold = threshold
		self.__feature = feature
		if self.__feature is None:
			raise ValueError("The feature has to be specified")


	def fit(self,X,y=None):
		return self


	def transform(self,X):

		binarizer = Binarizer(threshold=self.__threshold, copy=True)
		return binarizer.fit_transform(X[self.__feature].values)



class PCAtransformer(BaseEstimator, TransformerMixin):

	"""Class for performing dimensionality reduction via PCA"""

	def __init__(self, num_components):
		self.__num_components = num_components

	def fit(self,X,y=None):
		return self

	def transform(self, X):
		pca_scaler = PCA(n_components=self.__num_components).fit(X)
		inf = open(os.path.join(C.PATH, "pca_scaler.pkl"), "wb")
		pickle.dump(pca_scaler, inf); inf.close()
		X_new = pca_scaler.transform(X)
		columns = ["pca_{}".format(i) for i in range(X_new.shape[1])]
		try:
			return pd.DataFrame(X_new, columns=columns)
		except Exception as e:
			print("exception is ", e)



########################################################################################################################
#                           	CUSTOM
########################################################################################################################



class CustomTransformer(BaseEstimator, TransformerMixin):

	"""Custom transformation of the data by specifying the feature to transform and the function applied"""

	def __init__(self, feature, func=np.log1p):

		self.__func = func
		self.__feature = feature

	def fit(self,X,y=None):
		return self

	def transform(self,X):

		transformer = FunctionTransformer(func=self.__func, validate=True)
		return transformer.fit_transform(X[self.__feature].values)



########################################################################################################################
#                           				AUTO ML
########################################################################################################################


class AutoML:

	"""This class uses the TPOT module for performing automatic data preprocessing and feature engineering"""

	def __init__(self, X, y, method="reg"):
		from regressor_dict import regressor_config_dict 
		self.__method = method
		self.__y = y
		self.__X = X

		if self.__method not in ["reg", "clf"]:
			raise ValueError("method {} cannot be used, pick one of these: reg or clf ".format(self.__method))
		if self.__method == "clf":
			tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, cv=10)
		elif self.__method == "reg":
			tpot = TPOTRegressor(generations=1, population_size=50, verbosity=2, cv=10, n_jobs=30, scoring="r2")
		ola = tpot.fit(self.__X.values,self.__y.values)

		print(tpot.score(self.__X,self.__y))
		tpot.export(os.path.join(C.PATH, "tpot_results.py"))
		print(tpot.fitted_pipeline_)
		print("pareto ", tpot.pareto_front_fitted_pipelines_)



