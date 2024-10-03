import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm


if __name__ == '__main__':
    df_input = pd.read_csv("data/loan-approval/train.csv")

    df_input = df_input[["person_age","person_income","loan_amnt","loan_status"]].copy()

    # X sind die erklärenden Variablen (Features), y ist das Label (Zielvariable)
    X = df_input.drop(columns=["loan_status"])
    y = df_input["loan_status"]

    # Stratified K-Fold Cross Validation mit 10 Folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Beispielmodell, hier ein Random Forest
    model = RandomForestClassifier(random_state=42)

    # Initialize a progress bar using tqdm
    progress_bar = tqdm(total=skf.get_n_splits(), desc="Cross-Validation Progress", unit="fold")

    # List to store the cross-validation scores
    cv_scores = []

    # Manually iterate over the folds
    for train_index, test_index in skf.split(X, y):
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

    # Ausgabe der Kreuzvalidierungsergebnisse
    print("Durchschnittliche Kreuzvalidierungsergebnisse: ", sum(cv_scores) / len(cv_scores))
    print("Einzelergebnisse: ", cv_scores)