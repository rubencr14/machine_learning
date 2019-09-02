import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from Pipelines import pipelines
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import Configuration as C
import scipy, os, pickle
from scipy.stats import randint as sp_randint
import pandas as pd
from sklearn.model_selection import cross_validate
from FeatureEngineering import FeatureSelectionRFE, FeatureScaling, BoxCoxTransform,\
							YeoJohnsonTransform, LogTransform, PowerTransformationsWrapper, AutoML, CreatePolynomials, FeatureSelectionRFECV, PCAtransformer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB

__all__=["Hypertuning"]
__author__ = "Ruben Canadas Rodriguez"
__mail__ = "rubencr14@gmail.com"
__version__ = 1.0

#TODO: Apply parameter tuning for other methods also!

class ModelParameters(object):

	"""This class contains the different ranges of values that parameters can adopt"""

	def __init__(self, X, y, search):
		self._search = search
		self._features = len(X.columns.values)


	def extra_tree_classifier(self):
		
		params_grid = {  "n_estimators": range(10,100,10), 
					"criterion": ["gini", "entropy"], 
					"max_features": range(20, 100, 5),
					"min_samples_leaf": range(20,50,5),
					"min_samples_split": range(15,36,5)
					} 

		params_random = {"n_estimators": sp_randint(10,100), 
					"criterion": ["gini", "entropy"], 
					"max_features": sp_randint(10, self._features),
					"min_samples_leaf": sp_randint(20,50),
					"min_samples_split": sp_randint(15,45)
					} 					

		if self._search == "random": return params_random
		elif self._search == "grid": return params_grid

	def support_vector_machine_classifier(self):
		
		params_grid = { "kernel": ["linear", "sigmoid", "rbf"],
					"C":np.arange(1,10,0.1),
					"gamma": np.logspace(-4, 4, 100),
					"class_weight": ["balance", None]
		}
		
		params_random = { "kernel": ["linear", "sigmoid", "rbf"],
					"C":scipy.stats.uniform(1,3),
					"gamma": scipy.stats.uniform(0.1, 10),
					"class_weight": ["balanced", None],

		}
			
		if self._search == "random": return params_random
		elif self._search == "grid": return params_grid

	def voting_classifier(self):

		params_random = {"svc__kernel": ["linear", "sigmoid", "rbf"],
					"svc__C":scipy.stats.uniform(1,3),
					"svc__gamma": scipy.stats.uniform(0.1, 10),
					"svc__class_weight": ["balanced", None],
					"xt__n_estimators": sp_randint(10,100), 
					"xt__criterion": ["gini", "entropy"], 
					"xt__max_features": sp_randint(10, self._features),
					"xt__min_samples_leaf": sp_randint(20,50),
					"xt__min_samples_split": sp_randint(15,45)
		}
		
		return params_random



class Hypertuning(ModelParameters):

	"""This class inherits the parameters from its superclass ModelParameters in order to
	perform the grid/random search """

	def __init__(self, X, y, model, cv=10, verbose=True, search="random"):
		super(Hypertuning, self).__init__(X, y, search)
		self._model = model
		self._model_type = C.TYPE
		self._n_jobs = 20
		self.__cv = cv
		self.__verbose = verbose
		if self._model_type == "clf":
			self.__scoring = "accuracy"
		elif self._model_type == "reg":
			self.__scoring = "r2"
		if self._model =="extra_tree_classifier":
			self._params = self.extra_tree_classifier()
		elif self._model == "support_vector_machine_classifier":
			self._params = self.support_vector_machine_classifier()
		elif self._model == "voting":
			self._params = self.voting_classifier()
		else:
			raise ValueError("Method does not exist!")

	def parameter_search(self, X, y):


		if self._search == "grid":
			grid = GridSearchCV(param_grid=self._params, estimator=SVC(), scoring=self.__scoring, cv=self.__cv)
			grid.fit(X, y)
			if self.__verbose:
				print("Best parameters: {} with an score of {}".format(grid.best_params_, grid.best_score_))
			return grid_result.best_params_
		elif self._search == "random":
			grid = RandomizedSearchCV(param_distributions=self._params, estimator=VotingClassifier(estimators=[("svc", SVC(probability=True)), ("xt", ExtraTreesClassifier()), ("bn", BernoulliNB())], voting="soft", weights=[2,2,1]), 
										scoring=self.__scoring, cv=self.__cv, verbose=3, n_iter=7000, n_jobs=self._n_jobs)
			grid.fit(X, y)
			if self.__verbose:
				print("Best parameters: {} with an score of {}".format(grid.best_params_, grid.best_score_))
				df = pd.DataFrame().from_dict(grid.cv_results_)
				df.to_csv("grid_results.csv", sep=",")
			return grid.best_params_

################################################################################################################################################
#
#            									THIS PART BELOW IS FOR TESTING!
###############################################################################################################################################

if __name__ == "__main__":

	to_drop = []
	combination = ["BoxCox", "std"]
	train_file = open(os.path.join(C.PATH, "train.pkl"), "rb")
	train = pickle.load(train_file); train_file.close()
	X, y = train.drop([C.TARGET], axis=1), train[C.TARGET]
	X_final = pipelines(X, y, combination, to_drop, pipe="pipe_10")
	#print("X_final ", X_final.columns.values); exit()
	#tune = Hypertuning(X_final, y, model="voting", search="random")
	#best_params = tune.parameter_search(X_final, y)

	from sklearn.externals import joblib
	#model = SVC(C=best_params["C"], gamma=best_params["gamma"], class_weight=best_params["class_weight"], kernel=best_params["kernel"])
	model = VotingClassifier(estimators=[("svc", SVC(probability=True, C=2.1417, gamma=0.12204, kernel="rbf", class_weight="balanced")), ("xt", ExtraTreesClassifier(criterion="entropy", max_features=17, min_samples_leaf=20, min_samples_split=26, n_estimators=32)), ("bn", BernoulliNB())], voting="hard", weights=[5,4,0])
	model.fit(X_final, y)
	joblib.dump(model, os.path.join(C.PATH, "model.pkl"))


	def open_test():
		from sklearn.externals import joblib
		inf = open(os.path.join(C.PATH, "val.pkl"), "rb")
		test = pickle.load(inf); inf.close()
		scale_file = open(os.path.join(C.PATH, "std_scaler_pipe_10_1.pkl"), "rb")
		scaler = pickle.load(scale_file); scale_file.close()
		X_test, y_test = test.drop([C.TARGET], axis=1), test[C.TARGET]
		columns = X_test.columns.values
		X_test = scaler.transform(X_test)
		X_test = pd.DataFrame(X_test, columns=columns)
		pca_file = open(os.path.join(C.PATH, "pca_scaler.pkl"), "rb")
		pca_scaler = pickle.load(pca_file); pca_file.close()
		X_test = pca_scaler.transform(X_test)
		pca_columns = ["pca_{}".format(i) for i in range(X_test.shape[1])]
		X_test = pd.DataFrame(X_test, columns=pca_columns)
		columns_file = open(os.path.join(C.PATH, "columns_pipe_10_SBS.pkl"), "rb")
		to_drop = pickle.load(columns_file); columns_file.close()
		print("drop ", len(to_drop))
		for elem in pca_columns:
			if elem not in to_drop:
				X_test = X_test.drop(elem, axis=1)
		yeo = PowerTransformationsWrapper(method="BoxCox")
		X_test = yeo.fit_transform(X_test)
		model = joblib.load(os.path.join(C.PATH, "model.pkl"))
		y_pred = model.predict(X_test)
		print("sdfsdf ", confusion_matrix(y_test, y_pred))
		#results = cross_validate(model, X_test, y_test, scoring="accuracy", cv=10, n_jobs=10)

		#X_test = pipelines(X_test, y_test, combination, to_drop, pipe="pipe_5")
		return X_test, y_test
open_test()