#%%
# Imports
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, LassoLars, SGDRegressor, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

#%%
# Define models
models_dict = {
    'lasso': Lasso(),
    'lassolars': LassoLars(),
    'elasticnet': ElasticNet(),
    'sgd_lasso': SGDRegressor(),
    'sgd_elasticnet': SGDRegressor(),
    'regression_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor(),
    'gradient_boosting': GradientBoostingRegressor(),
    'xgboost': XGBRegressor()
}

# Models that need standardization
needs_standardization = ['lasso', 'lassolars', 'elasticnet', 'sgd_lasso', 'sgd_elasticnet']

# Wrap models that need standardization in a pipeline with a StandardScaler
for name in needs_standardization:
    model = models_dict[name]
    models_dict[name] = Pipeline([
        ('scaler', StandardScaler()),  # Apply standardization
        ('regressor', model)           # Fit the regressor
    ])

# Define hyperparameter grids, adjusting for pipeline
hyperparameter_grids = {
    'lasso': {'regressor__alpha': [0.01, 0.1, 1, 10, 100]},
    'lassolars': {'regressor__alpha': [0.01, 0.1, 1, 10, 100]},
    'elasticnet': {
        'regressor__alpha': [0.01, 0.1, 1, 10, 100],
        'regressor__l1_ratio': [0.2, 0.5, 0.8]
    },
    'sgd_lasso': {
        'regressor__penalty': ['l1'],
        'regressor__alpha': [0.00001, 0.001, 0.001, 0.01, 0.1, 1, 10],
        'regressor__loss': ['epsilon_insensitive'],
        'regressor__learning_rate': ['constant', 'optimal', 'invscaling'],
        'regressor__eta0': [0.001, 0.01, 0.1]
    },
    'sgd_elasticnet': {
        'regressor__penalty': ['elasticnet'],
        'regressor__alpha': [0.00001, 0.001, 0.001, 0.01, 0.1, 1, 10],
        'regressor__l1_ratio': [0.01, 0.1, 0.2, 0.5],
        'regressor__loss': ['epsilon_insensitive'],
        'regressor__learning_rate': ['constant', 'optimal', 'invscaling'],
        'regressor__eta0': [0.001, 0.01, 0.1]
    },
    'regression_tree': {
        'max_depth': [3, 5, 10, 20, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10, 50],
        'max_features': [None, 'sqrt', 'log2']
    },
    'random_forest': {
        'n_estimators': [100],
        'max_depth': [3, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [1, 10],
        'max_features': [None, 'sqrt', 'log2']
    },
    'gradient_boosting': {
        'n_estimators': [100],
        'max_depth': [3, 5],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'learning_rate': [0.1]
    },
    'xgboost': {
        'max_depth': [6],
        'min_child_weight': [1],
        'subsample': [0.5, 1],
        'colsample_bytree': [1],
        'gamma': [0],
        'eta': [0.3],
        'alpha': [0, 1],
        'tree_method': ['auto']
    }
}

def process_dataset(train_file_path, test_file_path, hyperparameter_grids):
    # Load training and test data
    train_df = pd.read_csv(train_file_path, index_col=0)  # Assuming participant ID is the index
    test_df = pd.read_csv(test_file_path, index_col=0)  # Assuming participant ID is the index
    
    # Split features and target
    X_train = train_df.drop('EPDS_TOT_8wPP', axis=1)
    y_train = train_df['EPDS_TOT_8wPP']
    X_test = test_df.drop('EPDS_TOT_8wPP', axis=1)
    y_test = test_df['EPDS_TOT_8wPP']
    
    # Store participant IDs (the index) for test data
    participant_ids = test_df.index
    
    results = {}
    for model_name, model in models_dict.items():
        if model_name in hyperparameter_grids:
            grid_search = GridSearchCV(model, hyperparameter_grids[model_name], cv=10, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Training MSE and R-squared
            best_train_predictions = best_model.predict(X_train)
            train_mse = mean_squared_error(y_train, best_train_predictions)
            train_r2 = r2_score(y_train, best_train_predictions)
            
            # Test MSE and R-squared
            predictions = best_model.predict(X_test)
            test_mse = mean_squared_error(y_test, predictions)
            test_r2 = r2_score(y_test, predictions)

            # Initialize storage for feature importances and non-zero features
            feature_importances = None
            non_zero_features = None
            
            # Capture feature importances if the model provides them
            if hasattr(best_model, 'feature_importances_'):
                feature_importances = dict(zip(X_train.columns, best_model.feature_importances_))

            # Identify non-zero features for models with coef_ attribute
            coefs = None
            if hasattr(best_model, 'coef_'):
                coefs = best_model.coef_
            elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['regressor'], 'coef_'):
                coefs = best_model.named_steps['regressor'].coef_

            if coefs is not None:
                if len(coefs) > 0 and coefs.ndim > 1:
                    coefs = coefs[0]
                non_zero_features = [X_train.columns[i] for i, coef in enumerate(coefs) if coef != 0]
                feature_coefficients = dict(zip(X_train.columns, coefs))  # Map coefficients to features
            
            # Store the results including the best model and features info
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'train_MSE': train_mse,
                'test_MSE': test_mse,
                'train_R2': train_r2,
                'test_R2': test_r2,
                'feature_importances': feature_importances,
                'coefficients': coefs.tolist() if coefs is not None else None,
                'feature_coefficients': feature_coefficients if coefs is not None else None,
                'non_zero_features': non_zero_features,
                'best_model': best_model,
                'y_test': y_test.tolist(),
                'participant_ids': participant_ids.tolist(),  # Store participant IDs
                'feature_names': X_train.columns.tolist(),
                'predictions': predictions.tolist(),
            }

    return results

def save_results_with_pickle(results, filename='model_results.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

# WEEK 12
#%%
train_file_path_k3 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week12_train_knn_imputed_k3.csv"
test_file_path_k3 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week12_test_knn_imputed_k3.csv"
results_k3 = process_dataset(train_file_path_k3, test_file_path_k3, hyperparameter_grids)

#%%
save_results_with_pickle(results_k3, filename='model_results_k3.pkl')

#%%
train_file_path_k5 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week12_train_knn_imputed_k5.csv"
test_file_path_k5 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week12_test_knn_imputed_k5.csv"
results_k5 = process_dataset(train_file_path_k5, test_file_path_k5, hyperparameter_grids)

#%%
save_results_with_pickle(results_k5, filename='model_results_k5.pkl')

#%%
train_file_path_k7 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week12_train_knn_imputed_k7.csv"
test_file_path_k7 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week12_test_knn_imputed_k7.csv"
results_k7 = process_dataset(train_file_path_k7, test_file_path_k7, hyperparameter_grids)

#%%
save_results_with_pickle(results_k7, filename='model_results_k7.pkl')

# WEEK 20
#%%
train_file_path_k3 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week20_train_knn_imputed_k3.csv"
test_file_path_k3 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week20_test_knn_imputed_k3.csv"
results_k3 = process_dataset(train_file_path_k3, test_file_path_k3, hyperparameter_grids)

#%%
save_results_with_pickle(results_k3, filename='model_results_k3_week20.pkl')

#%%
train_file_path_k5 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week20_train_knn_imputed_k5.csv"
test_file_path_k5 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week20_test_knn_imputed_k5.csv"
results_k5 = process_dataset(train_file_path_k5, test_file_path_k5, hyperparameter_grids)

#%%
save_results_with_pickle(results_k5, filename='model_results_k5_week20.pkl')

#%%
train_file_path_k7 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week20_train_knn_imputed_k7.csv"
test_file_path_k7 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week20_test_knn_imputed_k7.csv"
results_k7 = process_dataset(train_file_path_k7, test_file_path_k7, hyperparameter_grids)

#%%
save_results_with_pickle(results_k7, filename='model_results_k7_week20.pkl')

# WEEK 28
#%%
train_file_path_k3 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week28_train_knn_imputed_k3.csv"
test_file_path_k3 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week28_test_knn_imputed_k3.csv"
results_k3 = process_dataset(train_file_path_k3, test_file_path_k3, hyperparameter_grids)

#%%
save_results_with_pickle(results_k3, filename='model_results_k3_week28.pkl')

#%%
train_file_path_k5 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week28_train_knn_imputed_k5.csv"
test_file_path_k5 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week28_test_knn_imputed_k5.csv"
results_k5 = process_dataset(train_file_path_k5, test_file_path_k5, hyperparameter_grids)

#%%
save_results_with_pickle(results_k5, filename='model_results_k5_week28.pkl')

#%%
train_file_path_k7 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week28_train_knn_imputed_k7.csv"
test_file_path_k7 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\week28_test_knn_imputed_k7.csv"
results_k7 = process_dataset(train_file_path_k7, test_file_path_k7, hyperparameter_grids)

#%%
save_results_with_pickle(results_k7, filename='model_results_k7_week28.pkl')

# POSTPARTUM
#%%
train_file_path_k3 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\pp_train_knn_imputed_k3.csv"
test_file_path_k3 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\pp_test_knn_imputed_k3.csv"
results_k3 = process_dataset(train_file_path_k3, test_file_path_k3, hyperparameter_grids)

#%%
save_results_with_pickle(results_k3, filename='model_results_k3_pp.pkl')

#%%
train_file_path_k5 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\pp_train_knn_imputed_k5.csv"
test_file_path_k5 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\pp_test_knn_imputed_k5.csv"
results_k5 = process_dataset(train_file_path_k5, test_file_path_k5, hyperparameter_grids)

#%%
save_results_with_pickle(results_k5, filename='model_results_k5_pp.pkl')

#%%
train_file_path_k7 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\pp_train_knn_imputed_k7.csv"
test_file_path_k7 = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette\pp_test_knn_imputed_k7.csv"
results_k7 = process_dataset(train_file_path_k7, test_file_path_k7, hyperparameter_grids)

#%%
save_results_with_pickle(results_k7, filename='model_results_k7_pp.pkl')