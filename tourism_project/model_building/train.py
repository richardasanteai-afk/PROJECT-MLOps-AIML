# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-training-experiment")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# One-hot encode all categorical columns and scale numeric features
numeric_features = ['Age',
       'CityTier', 'DurationOfPitch',
       'NumberOfPersonVisiting', 'NumberOfFollowups',
       'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
       'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
       'MonthlyIncome']

categorical_features = ['TypeofContact','Occupation', 'Gender','ProductPitched',
                        'MaritalStatus','Designation']

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, 
eval_metric='logloss',n_jobs=-1)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100], # Number of boosting trees. More trees can improve performance but increase training time.
    'xgbclassifier__max_depth': [4, 6], # Maximum depth of each tree. Higher values increase model complexity and risk of overfitting.
    'xgbclassifier__learning_rate': [0.05, 0.1], # Step size used during boosting. Smaller values may improve generalization but require more trees.
    'xgbclassifier__colsample_bytree': [0.7], # Fraction of features sampled when building each tree.
    'xgbclassifier__subsample': [0.8], # Fraction of training samples randomly sampled for each tree. Adds row-level randomness and further helps prevent overfitting.
    'xgbclassifier__reg_lambda': [1.0], # L2 regularization strength. Higher values help reduce overfitting.
    'xgbclassifier__min_child_weight': [1]   # Minimum sum of instance weight (hessian) needed in a child node.     
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
    )
    grid_search.fit(Xtrain, ytrain)

    # Log parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    classification_threshold = 0.35

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)

    train_report = classification_report(
        ytrain, y_pred_train, output_dict=True
    )
    test_report = classification_report(
        ytest, y_pred_test, output_dict=True
    )

    print(classification_report(ytest, y_pred_test))

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
    })

    model_path = (
        "tourism_project/deployment/"
        "best_tourism_package_model_v1.joblib"
    )
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
