# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 17:20:00 2018

@author: Daniel
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold, cross_val_score
   
def get_car_data():
    
    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "saftey", "condition"]
    
    # Read the data from the .csv file
    data_df = pd.read_csv("car_data.csv", header=None, names=col_names)

    # Convert categorical entries into numerical values
    cleanup_nums = {"buying": {"vhigh": 3., "high": 2., "med": 1., "low": 0.}, 
                    "maint": {"vhigh": 3., "high": 2., "med": 1., "low":0.}, 
                    "doors": {"2": 2., "3": 3., "4": 4., "5more": 5.}, 
                    "persons": {"2": 2., "4": 4., "more": 5.}, 
                    "lug_boot": {"small": 0., "med": 1., "big": 2.}, 
                    "saftey": {"low": 0., "med": 1., "high": 2.},
                    "condition": {"unacc": 0., "acc": 1., "good": 2., "vgood": 3.}}  
    
    data_df.replace(cleanup_nums, inplace=True)
    data_df.head()

    # Separate the data
    data_cols = col_names[0:6]
    target_col = ["condition"]
    
    # Convert the data into np arrays
    data = np.array(data_df[data_cols])
    targets = np.array(data_df[target_col])
 
    return data, targets

def get_autism_data():
    
    col_names = ["A1_score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", 
                 "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
                 "age", "gender", "ethnicity", "jundice", "autism", 
                 "country_of_res", "used_app_before", "result", "age_desc", 
                 "relation", "Class/ASD"]
    
    # Read the .csv containing the data
    data_df = pd.read_csv("Autism-Adult-Data.csv", header=None, names=col_names, na_values="?")

    # Convert the categorical columns into one-hots
    one_hot_cols = ["ethnicity", "country_of_res", "age_desc", "relation"]
    new_data_df = pd.get_dummies(data_df, columns=one_hot_cols)
    
    # Change the binary labels into 1s or 0s
    cleanup_nums = {"gender": {"f": 0., "m": 1.},
                    "jundice": {"no": 0., "yes": 1.},
                    "autism": {"no": 0., "yes": 1.},
                    "used_app_before": {"no": 0., "yes": 1.},
                    "Class/ASD": {"NO": 0., "YES": 1.}}
    new_data_df.replace(cleanup_nums, inplace=True)
    
    # Fill any missing values with the most frequent value in that column
    fill_NaN = Imputer(missing_values=np.nan, strategy='most_frequent', axis=1)
    imputed_df = pd.DataFrame(fill_NaN.fit_transform(new_data_df))
    imputed_df.columns = new_data_df.columns
    imputed_df.index = new_data_df.index

    # Split the data and the targets and numpy arrays
    data = np.array(imputed_df.loc[:, imputed_df.columns != "autism"])
    targets = np.array(imputed_df["autism"])
    
    return data, targets

def get_mpg_data():
    
    col_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", 
                 "acceleration", "model year", "origin", "car name"]
    
    # Read the .csv containing the data
    data_df = pd.read_csv("auto-mpg_data.csv", delim_whitespace=True, 
                          header=None, names=col_names, na_values="?")
    
    # Convert the categorical columns into one-hots
    one_hot_cols = ["car name"]
    new_data_df = pd.get_dummies(data_df, columns=one_hot_cols)
    
    # Fill any missing values with the mean value in that column
    fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
    imputed_df = pd.DataFrame(fill_NaN.fit_transform(new_data_df))
    imputed_df.columns = new_data_df.columns
    imputed_df.index = new_data_df.index
    
    # Split the data and the targets and numpy arrays
    data = np.array(imputed_df.loc[:, imputed_df.columns != "mpg"])
    targets = np.array(imputed_df["mpg"])
    
    return data, targets

def main(): 
    # Load the dataset
    print("Loading the data...")
    isRegression = False
    
    # Uncomment one of the following functions for the desired data
#    data, targets = get_car_data()
#    data, targets = get_autism_data()

    # Since this is regression data, uncomment the isRegression = True line
    # and comment out the data = preprocessing.scale(data) line. I don't think 
    # it should be scaled for continuous data - and the accuracy shows that
    data, targets = get_mpg_data()
    isRegression = True # uncomment this line for mpg data

    # Normalize the data
    std_scaler = preprocessing.StandardScaler().fit(data)
#    data = preprocessing.scale(data)
     
    # Get the predicted targets
    print("Testing...")
          
    # Obtain a classifier
    k_neighbors = 9
    if not isRegression:
        classifier = KNeighborsClassifier(n_neighbors = k_neighbors)
    else:
        classifier = KNeighborsRegressor(n_neighbors = k_neighbors)

    # Obtain the accuracy of the classifier
    # Configure k-fold validation
    k_fold = KFold(len(np.ravel(targets)), n_folds=10, shuffle=True, random_state=18)
    accuracy_score = cross_val_score(classifier, data, np.ravel(targets), cv=k_fold, n_jobs=1).mean()
    
    # Display the accuracy results
    print("Total Accuracy: {:.2f}%".format(accuracy_score * 100.0))

main()