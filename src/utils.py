import os
import sys
import pandas as pd 
import numpy as np
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """Saves a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object should be saved.
        obj: The Python object to be saved.

    Raises:
        CustomException: If there is an error during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

    
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error occurred while saving object at {file_path}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            # Using GridSearchCV to find the best parameters
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            # Train the model
            model.fit(X_train, y_train)

            # Predicting the test set results
            y_test_pred = model.predict(X_test)

            # Getting the r2 score for the model
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
    
        return report
    except Exception as e:
        logging.error("Error occurred while evaluating models")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)