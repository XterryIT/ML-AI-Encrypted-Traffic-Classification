import numpy as np
import pandas as pd
import time
import joblib
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Set print options to suppress scientific notation for readability
#np.set_printoptions(suppress=True)

def prepare_data():
    try:
        df = pd.read_csv('data/sample.csv')
    except FileNotFoundError:
        print("ERROR: File not found!")
        return

    # fill any missing values (NaN) with 0
    df.fillna(0, inplace=True)

    # separating label and statistics
    x = df.drop(['Label'], axis=1)
    y = df['Label']

    # print the class distribution
    print(y.value_counts().sort_index())
    print("(0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)\n")

    # split data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=300)

    # preparing data for stage 1
    y_train_s1 = y_train.copy().replace({2: 1}) # replacing doh classes for one doh class
    # preparing data for stage 2
    mask_s2 = (y_train == 1) | (y_train == 2) # only use training rows where the original label was 1 or 2
    x_train_s2 = x_train[mask_s2]
    y_train_s2 = y_train[mask_s2]

    return [x_train, x_train_s2, x_test, y_train_s1, y_train_s2, y_test, x]

def feature_importance(model, feature_names, stage):
    importances = None

    # for tree-based models (Random Forest, Decision Tree)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'): # for linear models (Logistic Regression, LDA, SGDClassifier)
        importances = np.mean(np.abs(model.coef_), axis=0)

    # if importance scores were found
    if importances is not None:
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})  # create a DataFrame (combine feature names with their importance scores)
        fi_df = fi_df.sort_values(by='Importance', ascending=False) # sort from most important to the least important
        print(f"Top 10 features used by the model for {stage}:"); print(fi_df.head(10)) # print Top 10 most important features
    else:
        print("Feature Importance is not natively available or easily interpretable for this model.")

def roc_curve_plot(model_s1, model_s2, x_test, y_test):
    model_name = model_s1.__class__.__name__

    # check if the model has a decision_function or predict_proba method
    if not (hasattr(model_s1, 'predict_proba')):
        print(f"Skipping ROC plot: Model {model_name} lacks 'predict_proba' or 'decision_function'.")
        return
    
    y_test_s1 = y_test.copy().replace({2: 1}) # stage 1: actual binary Labels (0 vs 1/2)
    mask_s2 = (y_test == 1) | (y_test == 2) 
    x_test_s2 = x_test[mask_s2]
    y_test_s2 = y_test[mask_s2] # stage 2: actual binary labels (1 vs 2) - filter test set
    
    plt.figure(figsize=(12, 5))

    # subplot 1: stage 1 roc
    plt.subplot(1, 2, 1)
    # getting the prediction scores (probabilities or decision scores)
    if hasattr(model_s1, "predict_proba"):
        y_score_s1 = model_s1.predict_proba(x_test)[:, 1]
        fpr1, tpr1, _ = roc_curve(y_test_s1, y_score_s1)
        roc_auc1 = auc(fpr1, tpr1)
        plt.plot(fpr1, tpr1, label=f'Stage 1 (AUC = {roc_auc1:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} Stage 1: DoH vs Non-DoH')
    plt.legend(loc="lower right")

    # subplot 2: stage 2 ROC
    plt.subplot(1, 2, 2)
    if hasattr(model_s2, "predict_proba") and not x_test_s2.empty:
        y_score2 = model_s2.predict_proba(x_test_s2)[:, 1]
        # Map labels to 0 and 1 for the ROC function (1->0, 2->1)
        fpr2, tpr2, _ = roc_curve(y_test_s2, y_score2, pos_label=2)
        roc_auc2 = auc(fpr2, tpr2)
        plt.plot(fpr2, tpr2, color='orange', label=f'Stage 2 (AUC = {roc_auc2:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} Stage 2: Benign vs Malicious')
    plt.legend(loc="lower right")
    plt.show()

def model_training(model_s1, model_s2, x_train, x_train_s2, x_test, y_train, y_train_s2, y_test, feature_names):
    """
    Stage 1: Class 0(Non-DoH) vs {Class 1, 2} (DoH)
    Stage 2: Class 1 (Benign) vs Class 2 (Malicious)
    """

    start_time = time.time() # starting timer
    model_s1.fit(x_train, y_train) # training the first model (DoH vs Non-DoH)
    model_s2.fit(x_train_s2, y_train_s2) # training the second model (Benign vs Malicious)
    end_time = time.time() # stopping the timer

    # printitng out model name for visibility
    model_name = model_s1.__class__.__name__
    feature_list = list(feature_names)

    joblib.dump(model_s1, f"models/hierarchy/{model_name}_s1_multiclass.joblib") # Save the TRAINED OBJECT
    joblib.dump(model_s2, f"models/hierarchy/{model_name}_s2_multiclass.joblib")
    joblib.dump(feature_list, f"models/hierarchy/{model_name}_features.joblib") # Save feature names

    print(f"SUCCESS: Model saved as '{model_name}_s[1/2]_multiclass.joblib'")
    print(f"SUCCESS: Feature list saved as '{model_name}_features.joblib'")

    print(f"\n--- Results for: {model_name} ---") # printitng out model name for visibility

    # time spent on training
    training_time = end_time - start_time
    print(f"Time spent on training: {training_time:.2f}s")

    pred_s1 = model_s1.predict(x_test) # predict stage 1 (DoH or not?)
    y_pred = np.zeros(len(x_test), dtype=int) # create final prediction array (default to 0)
    
    # for every sample predicted as DoH (1), run stage 2 model
    doh_indices = np.where(pred_s1 == 1)[0]
    if len(doh_indices) > 0:
        x_test_doh = x_test.iloc[doh_indices]
        preds_s2 = model_s2.predict(x_test_doh)
        
        # map stage 2 results back to the final prediction list
        for i, idx in enumerate(doh_indices):
            y_pred[idx] = preds_s2[i]

    # setting matrix and report
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]) # set labels to ensure [0, 1, 2] order in the matrix)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%"); print("Confusion Matrix: (3x3)"); print(cm)
    print("\nClassification Report (0=NonDoH, 1=Benign, 2=Malicious):")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2],
                                target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']))

    # Feature Importance if supported
    if hasattr(model_s1, 'feature_importances_') or hasattr(model_s1, 'coef_'):
        feature_importance(model_s1, feature_names, "Stage 1 (Non-DoH vs DoH)")
        feature_importance(model_s2, feature_names, "Stage 2 (Benign vs Malicious)")
    else:
        print("\nSkipping Feature Importance (Not applicable to this model class).")

    roc_curve_plot(model_s1, model_s2, x_test, y_test)
    print("-" * 100)

def main():

    data = prepare_data() # preparing data
    # data has x_train, x_train_s2, x_test, y_train_s1, y_train_s2, y_test, x in given order

    # defining classifiers
    models = [DecisionTreeClassifier(random_state=42),
              RandomForestClassifier(n_estimators=100, random_state=42),
              KNeighborsClassifier(n_neighbors=10)]

    for model in models:
        model_s1 = clone(model)
        model_s2 = clone(model)
        model_training(model_s1, model_s2, data[0], data[1], data[2], data[3], data[4], data[5], data[6].columns) # training and printing results out

if __name__ == "__main__":
    main()
