import os
import sys
from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        import dill
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]
            # Special handling for CatBoost estimators which may not implement
            # __sklearn_tags__ and so break sklearn's GridSearchCV/is_classifier checks.
            model_module = getattr(model.__class__, "__module__", "").lower()
            model_name = model.__class__.__name__.lower()
            if "catboost" in model_module or "catboost" in model_name:
                # Manual grid search using KFold to avoid sklearn tag checks
                # Create fresh unfitted model instances for CV and final fit
                best_score = -float("inf")
                best_params = None
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                base_params = model.get_params() if hasattr(model, "get_params") else {}
                for p in ParameterGrid(para):
                    params_combined = {**base_params, **p}
                    # Try to instantiate a fresh model with combined params
                    try:
                        temp_model = model.__class__(**params_combined)
                    except Exception:
                        # fallback: try deepcopy and set params (skip if not possible)
                        try:
                            import copy
                            temp_model = copy.deepcopy(model)
                            temp_model.set_params(**p)
                        except Exception:
                            continue
                    cv_scores = []
                    for train_idx, val_idx in kf.split(X_train):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]
                        temp_model.fit(X_tr, y_tr)
                        preds = temp_model.predict(X_val)
                        cv_scores.append(r2_score(y_val, preds))
                    mean_score = float(np.mean(cv_scores)) if cv_scores else -float("inf")
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = p
                # Build a fresh final model instance with best params (if any)
                if best_params is not None:
                    final_params = {**base_params, **best_params}
                    model = model.__class__(**final_params)
                else:
                    # ensure model is an unfitted instance before final fit
                    try:
                        model = model.__class__(**(model.get_params() if hasattr(model, "get_params") else {}))
                    except Exception:
                        import copy
                        model = copy.deepcopy(model)
                model.fit(X_train, y_train)
            else:
                gs = GridSearchCV(model, para, cv=5, n_jobs=-1, verbose=0, refit=True)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        import dill
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)