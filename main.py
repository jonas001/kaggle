import pandas as pd
from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb


def train(folds, X, y, model):
    # Initialize a progress bar using tqdm
    progress_bar = tqdm(total=skf.get_n_splits(), desc="Cross-Validation Progress", unit="fold")

    # List to store the cross-validation scores
    cv_scores = []

    # Manually iterate over the folds
    for train_index, test_index in folds.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model on the training fold
        model.fit(X_train, y_train)

        # Make predictions on the test fold
        y_pred = model.predict(X_test)

        # Calculate accuracy for this fold
        score = accuracy_score(y_test, y_pred)
        cv_scores.append(score)

        # Update progress bar
        progress_bar.update(1)

    # Close the progress bar after the loop is complete
    progress_bar.close()

    #print("Durchschnittliche Kreuzvalidierungsergebnisse: ", sum(cv_scores) / len(cv_scores))

    # Ausgabe der Kreuzvalidierungsergebnisse
    return round(min(cv_scores), 3), round(max(cv_scores), 3), round(sum(cv_scores) / len(cv_scores), 3)


if __name__ == '__main__':
    print("load data")

    # read training data
    df_train = pd.read_csv("data/loan-approval/train.csv", index_col="id")

    # encode non numeric values
    df_train = pd.get_dummies(df_train, columns=['person_home_ownership','loan_intent', 'loan_grade'], drop_first=True, dtype=float)

    dict_default_on_file = {'Y': 1, 'N': 0}
    df_train['cb_person_default_on_file'] = df_train['cb_person_default_on_file'].map(dict_default_on_file)

    # Split data for training and evaluation
    X = df_train.drop(columns=["loan_status"])
    y = df_train["loan_status"]

    # Stratified K-Fold Cross Validation mit 10 Folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # initiate lists for results
    models = []
    model_name = []
    model_avg = []
    model_min = []
    model_max = []

    print("data loading finished")

    # DecisionTree
    print("Decision Tree: starting")
    time.sleep(1)
    model = DecisionTreeClassifier(random_state=42)
    acc_min, acc_max, acc_avg = train(skf, X, y, model)
    model_name.append("DecisionTree")
    models.append(model)
    model_avg.append(acc_avg)
    model_min.append(acc_min)
    model_max.append(acc_max)
    print("Decision Tree: finished")

    # Random Forest
    print("Random Forest: starting")
    time.sleep(1)
    model = RandomForestClassifier(random_state=42)
    acc_min, acc_max, acc_avg = train(skf, X, y, model)
    model_name.append("RandomForest")
    models.append(model)
    model_avg.append(acc_avg)
    model_min.append(acc_min)
    model_max.append(acc_max)
    print("Random Forest: finished")

    # Gradient Boosted Trees
    print("Gradient Boosted Trees: starting")
    time.sleep(1)
    model = GradientBoostingClassifier(random_state=42)
    acc_min, acc_max, acc_avg = train(skf, X, y, model)
    model_name.append("GradientBoostedTrees")
    models.append(model)
    model_avg.append(acc_avg)
    model_min.append(acc_min)
    model_max.append(acc_max)
    print("Gradient Boosted Trees: finished")

    # XGBoost
    print("XGBoost: starting")
    time.sleep(1)
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    acc_min, acc_max, acc_avg = train(skf, X, y, model)
    model_name.append("XGBoost")
    models.append(model)
    model_avg.append(acc_avg)
    model_min.append(acc_min)
    model_max.append(acc_max)
    print("XGBoost: finished")

    # lists to dataframe
    dict = {'name': model_name, 'avg': model_avg, 'min': model_min, 'max': model_max}
    df_result = pd.DataFrame(dict)
    print(df_result)

    # training best model
    print("Classification: starting")

    # read test data
    df_test = pd.read_csv("data/loan-approval/test.csv", index_col="id")

    # encode non numeric values
    df_test = pd.get_dummies(df_test, columns=['person_home_ownership','loan_intent', 'loan_grade'], drop_first=True, dtype=float)
    df_test['cb_person_default_on_file'] = df_test['cb_person_default_on_file'].map(dict_default_on_file)

    # find best model by highest average
    index_min = min(range(len(model_avg)), key=model_avg.__getitem__)
    model = models[index_min]

    # Train the model on the complete training data
    model.fit(X, y)

    # Make prediction
    df_test["loan_status"] = model.predict(df_test)

    df_test = df_test[["loan_status"]].copy()
    df_test.to_csv("data/loan-approval/submission.csv", index_label="id")

    print("Classification: finished")

