import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os
import Configuration as C
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split
except ImportError:
   exit("Sklearn could not be imported, install it via pip or conda")
import numpy as np
import pandas as pd
import errors, OutlierDetection

__author__ = "Ruben Canadas Rodriguez"
__mail__ = "rubencr14@gmail.com"
__version__ = 1.0


class DatasetSplitter(BaseEstimator, TransformerMixin):

    """This class splits the whole dataset into three sets: training, validation and test. The validation and test
    is for avoiding overfitting"""
    
    def __init__(self, to_drop=None):

        self.__random_state = 2019
        self.__path = C.PATH
        self.__to_drop = to_drop
        self.__y = None


    def __str__(self):
        
        return "{}".format(self.__df)

    @property
    def random_state(self):

        return self.__random_state

    @random_state.setter
    def random_state(self,value):

        self.__random_state = value

    def fit(self,X,y=None):
        """here X referes to the whole dataset and returns only X_train"""
        self.__y = y
        return self

    def transform(self,X,y=None):

        if not os.path.exists(os.path.join(C.PATH, "train.pkl")):
            if self.__to_drop is not None:
                X = X.drop(self.__to_drop, axis=1)
            columns = X.columns.values
            df = pd.concat([X,self.__y], axis=1)
            #df = OutlierRemover().fit_transform(df)
            train, test_and_validation = train_test_split(df, test_size=0.30, random_state=self.__random_state)
            validation, test = train_test_split(test_and_validation, test_size=0.60, random_state=self.__random_state) #Just to change the random state
            val_file = open(os.path.join(self.__path, "val.pkl"), "wb")
            pickle.dump(validation, val_file); val_file.close()
            test_file = open(os.path.join(self.__path, "test.pkl"), "wb")
            pickle.dump(test, test_file); test_file.close()
            train_file = open(os.path.join(self.__path, "train.pkl"), "wb")
            pickle.dump(pd.DataFrame(train,columns=df.columns.values), train_file); train_file.close()
            return pd.DataFrame(train, columns=columns)
        else:
            infile = open(os.path.join(C.PATH, "train.pkl"),"rb")
            train = pickle.load(infile); infile.close()
            columns = X.columns.values
            return pd.DataFrame(train, columns=columns)
        
class CategoricalAttributeImputer(BaseEstimator, TransformerMixin):

    """This class handles with missing values of categorical features using BaseEstimator and TransformerMixing
     super class attributes and methods derived from sklearn"""

    def __init__(self):
        self.__value_type = "most_frequent"

    def fit(self, X, y=None):
        return self


    def transform(self, X):

        missing_values = X.isnull().stack()[lambda x: x].index.tolist() # list of tuples of missing values (row, columns)
        output = X.copy() # saving in output variable a copy of X dataset
        columns_having_null_values = [] # Saving in this list container the name of columns having null values
        for column in missing_values:
            if column[1] not in columns_having_null_values:
                columns_having_null_values.append(column[1]) # Appending the columns having null values

        for col in columns_having_null_values:
            final = output[col].value_counts() # Returns a series specifying how many times each categ. feature repeated
            most_frequent_value = list(zip(final.index, final))[0][0] # Most frequent value of the column
            for null_element in missing_values:
                if null_element[1] == col:
                    output.loc[null_element[0], col] = most_frequent_value # Replacing null for most frequent
        return output


class AttributeImputer(BaseEstimator, TransformerMixin):

    """This class handles with missing values for either numerical/categorical features.
    This class uses the transformer CategoricalAttributeImputer in order to manage
    with categorical missing values using the most frequent categ. value"""


    def __init__(self, value_type="most_frequent"):

        self.__value_type = value_type
        self.__available = ["mean", "median", "most_frequent"]
        if self.__value_type not in self.__available:
            raise errors.MethodNotFoundError("Method: {} is not available, pick one of these: {}".format(self.__value_type,
                                                                                                 self.__available))

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        try:
            numerical_attributes, categorical_attributes = X.select_dtypes(include=["float64", "int64"]), \
                                                           X.select_dtypes(include=["object"]) # Separating numerical from categorical
            imputed_numerical_attributes = SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(pd.DataFrame(numerical_attributes))
            imputed_categorical_attributes = CategoricalAttributeImputer().fit_transform(categorical_attributes)
            return pd.concat([imputed_numerical_attributes, imputed_categorical_attributes], axis=1)

        except:
            print("NO MISSING values found, returning non-transformed data...")
            return X



class TransformCategoricalToNumerical(BaseEstimator, TransformerMixin):

    """This class handles with categorial variables. First we asssign an integer number to each label of the
    categorical features and then we use One-Hot enconding in order to create a feature for each categorical label
    (Check documentation for more on one-hot enconding or the Hands-On machine learning book by Aurelien Geron"""

    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X):

        df_feat = []
        numerical_attributes = X.select_dtypes(include=["float64", "int64"]) #Selecting numerical features
        categorical_attributes = X.select_dtypes(include=["object"]).copy() #Selecting categorical features
        if categorical_attributes is not None:
            categ_output = categorical_attributes.copy()
            for feat in categorical_attributes.columns.values:
                encoded, categ = categ_output[feat].factorize() #Converting labels into integers using factorize method from Pandas
                hot_encoder = OneHotEncoder(categories='auto')
                feat_1hot = hot_encoder.fit_transform(encoded.reshape(-1,1)).toarray()
                df_feat.append(pd.DataFrame(feat_1hot))
        try:
            df_categ = pd.concat(df_feat, axis=1)
            return df_categ

        except: #If no categorical features found return X without transforming
            print("NO CATEGORICAL values found, returning non-transformed data...")
            return X



class OutlierRemover(BaseEstimator, TransformerMixin):

    """This class uses the OutlierDetection method for removing those instances that seem to be outliers
    and that might be adding noise to the model"""

    def __init__(self, outlier_type="standard"):

        self.__outlier_type = outlier_type
        self.__outlier_types = ["standard","interquantile"]
        if self.__outlier_type not in self.__outlier_types:
            raise errors.MethodNotFoundError("Method {} not found, pick one of these: {}".format(self.__outlier_type,
                                                                                                 self.__outlier_types))

    def plot(self, lower, upper,df,x_field):

        ax = plt.subplot()
        ax.axvspan(min(df.loc[df[x_field] < lower][x_field]), max(df.loc[df[x_field] < lower][x_field]), alpha=0.2,
                   color="r")
        ax.axvspan(min(df.loc[df[x_field] > upper][x_field]), max(df.loc[df[x_field] > upper][x_field]), alpha=0.2,
                   color="r")
        sns.distplot(sorted(df[x_field]), kde=True, ax=ax)
        plt.title(x_field)
        plt.show()
        plt.clf()


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.__outlier_type == "interquantile":
            numerical_attributes = X.select_dtypes(include=["float64", "int64"])
            for column in numerical_attributes.columns.values:
                try:
                    outlier = OutlierDetection.InterQuantileRangeOutliersDetection(X, column)
                    lower, upper, outliers = outlier.OutlierDetectionInterquantile()
                    indices = outliers.index.values
                    #self.Plot(lower, upper, X, column)
                    X = X.drop(indices, axis=0)
       
                except ValueError:
                    print("no outliers found for: {} feature".format(column))
                    continue
            return X

        elif self.__outlier_type == "standard":
            numerical_attributes = X.select_dtypes(include=["float64", "int64"])
            for column in numerical_attributes.columns.values:
                try:
                    outlier = OutlierDetection.InterQuantileRangeOutliersDetection(X, column)
                    lower, upper, outliers = outlier.OutlierDetectionStandard()
                    indices = outliers.index.values
                    #self.Plot(lower, upper, X, column)
                    X = X.drop(indices, axis=0)
                except ValueError:
                    print("no outliers found for: {} feature".format(column))
                    continue
            return X

        else:
            pass


