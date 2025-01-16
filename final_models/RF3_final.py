import scanpy as sc
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cleanlab.classification import CleanLearning
from scipy.sparse import issparse
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# --- Load Data ---
if __name__ == '__main__':

    # --- Load Data ---
    reference_data = sc.read_h5ad("GSE125449_filtered.h5ad")
    target_data_1 = sc.read_h5ad("GSE189903_f_N1215_R15.h5ad")
    target_data_2 = sc.read_h5ad("GSE151530_f_N1218_R18.h5ad")

    # --- Find Common Genes ---
    common_genes = reference_data.var_names.intersection(target_data_1.var_names).intersection(target_data_2.var_names)
    reference_data = reference_data[:, common_genes]
    target_data_1 = target_data_1[:, common_genes]
    target_data_2 = target_data_2[:, common_genes]

    # --- Prepare Data ---
    def preprocess_data(data):
        X = data.X if issparse(data.X) else data.X.toarray()
        X = normalize(X, axis=1, norm='l2')
        return X

    X_ref = preprocess_data(reference_data)
    X_target_1 = preprocess_data(target_data_1)
    X_target_2 = preprocess_data(target_data_2)

    ref_labels = reference_data.obs['Type'].to_numpy()
    unique_labels = np.unique(ref_labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y_ref = np.array([label_mapping[label] for label in ref_labels])

    # --- Clean the Dataset with Cleanlab ---
    # Initialize a simple baseline model for Cleanlab
    rf_clf = RandomForestClassifier(random_state=42)
    cleaner = CleanLearning(rf_clf)
    cleaner.fit(X_ref, y_ref)
    cleaned_labels = cleaner.predict(X_ref)  # Reclassified labels based on model's confidence
    reference_data.obs['Type_cleaned'] = [unique_labels[label] for label in cleaned_labels]  # Update labels with reclassified ones
    label_comparison_df = pd.DataFrame({
        'Old Label': reference_data.obs['Type'],
        'New Label': reference_data.obs['Type_cleaned']
    })
    label_comparison_df.to_csv("label_comparison_6.csv", index=False)

    # --- 10-Fold Cross-Validation with Random Forest ---
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_ref, y_ref), 1):
        print(f"Fold {fold}...")

        # Split data
        x_train, x_test = X_ref[train_idx], X_ref[test_idx]
        y_train, y_test = y_ref[train_idx], y_ref[test_idx]

        # Build and Train Random Forest Model
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(x_train, y_train)

        model_filename = f"RF_final_2_Cleanlab_125449_fold_{fold}.pkl"
        joblib.dump(rf_clf, model_filename)

        # Evaluate Model
        y_pred = rf_clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)

    # Print Overall Accuracy
    overall_accuracy = np.mean(fold_accuracies)
    print(f"Overall Accuracy (10-Fold CV): {overall_accuracy * 100:.2f}%")

    # --- Predict and Save New Columns ---
    def add_predictions_to_adata(model, data, unique_labels, filename):
        X_data = preprocess_data(data)
        y_pred = model.predict(X_data)
        predicted_labels = [unique_labels[label] for label in y_pred]
        data.obs['GSE125449_RF'] = predicted_labels
        data.write_h5ad(filename)

    # Save Predictions for Target Data
    add_predictions_to_adata(rf_clf, target_data_1, unique_labels, "GSE189903_f_N1215_R1215.h5ad")
    add_predictions_to_adata(rf_clf, target_data_2, unique_labels, "GSE151530_f_N1218_R1218.h5ad")

    # --- Confusion Matrices ---
    def plot_confusion_matrix(y_true, y_pred, labels, filename):
        conf_matrix = confusion_matrix(y_true, y_pred)
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

        # Define a pastel blue colormap
        pastel_red_cmap = LinearSegmentedColormap.from_list(
        "pastel_red", ["#ffe0e0", "#ffb3b3", "#ff8080", "#ff4d4d", "#ff1a1a"]
        
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix_percent,
            annot=True,
            fmt='.2f',
            cmap=pastel_red_cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Percentage (%)'}
        )
        plt.title('Confusion Matrix - Percentages (True vs Predicted Labels)')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Plot Confusion Matrices for Target Data 1 and 2
    y_pred_1 = rf_clf.predict(X_target_1)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    y_pred_1_labels = [reverse_label_mapping[label] for label in y_pred_1]
    target_1_labels = target_data_1.obs['Type']
    y_true_1_labels = target_1_labels.to_numpy()
    plot_confusion_matrix(y_true_1_labels, y_pred_1_labels, unique_labels, "confusion_matrix_target_1_RF_6.png")

    y_pred_2 = rf_clf.predict(X_target_2)
    y_pred_2_labels = [reverse_label_mapping[label] for label in y_pred_2]
    target_2_labels = target_data_2.obs['Type']
    y_true_2_labels = target_2_labels.to_numpy()
    plot_confusion_matrix(y_true_2_labels, y_pred_2_labels, unique_labels, "confusion_matrix_target_2_RF_6.png")
