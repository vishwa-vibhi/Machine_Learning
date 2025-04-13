!pip install lime shap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector

import lime
import lime.lime_tabular
import shap
# ---------- A1 FUNCTIONS ----------
def load_dataset(filepath: str) -> pd.DataFrame:
    """Load Excel dataset into a pandas DataFrame."""
    return pd.read_excel(filepath)

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix of features only (excluding target)."""
    feature_cols = [col for col in df.columns if col.startswith("embed_")]
    return df[feature_cols].corr()

def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    """Display the correlation heatmap of feature embeddings."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# ---------- A2 FUNCTIONS ----------

def bin_output_column(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Bin the continuous output values into discrete classification classes."""
    bins = np.linspace(0, 5, n_bins + 1)
    labels = list(range(n_bins))
    df["class"] = pd.cut(df["output"], bins=bins, labels=labels, include_lowest=True)
    df["class"] = df["class"].astype(int)
    return df

def split_features_labels(df: pd.DataFrame):
    """Split features (embeddings) and target (class)."""
    X = df[[col for col in df.columns if col.startswith("embed_")]]
    y = df["class"]
    return X, y

def apply_pca(X: pd.DataFrame, variance_threshold: float):
    """Apply PCA and reduce dimensions to retain 99% variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Select number of components to retain 99% variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    pca_final = PCA(n_components=n_components)
    X_reduced = pca_final.fit_transform(X_scaled)
    return X_reduced, pca_final

def train_classifiers(X_train, y_train, X_test, y_test):
    """Train and evaluate classification models."""
    results = {}

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results[name] = accuracy

    return results

#A4
# Apply Sequential Feature Selection using LR and RF
def apply_sequential_feature_selection(X: pd.DataFrame, y: pd.Series, k_features: int = 5, test_size: float = 0.2):
    # k_features : Number of features to select.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    selected_features = {}
    accuracies = {}

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    for name, model in models.items():
        # Create sequential feature selector with forward selection strategy
        selector = SequentialFeatureSelector(
            model,
            n_features_to_select=k_features,
            direction='forward',       # Adds one feature at a time
            scoring='accuracy',
            cv=5                       # Use 5-fold cross-validation to evaluate subsets
        )
        
        # Fit selector to data
        selector.fit(X_train, y_train)

        # Get boolean mask and extract selected column names
        selected_cols = X_train.columns[selector.get_support()].tolist()
        selected_features[name] = selected_cols

        # Train model with selected features
        X_train_selected = X_train[selected_cols]
        X_test_selected = X_test[selected_cols]
        
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        # Evaluate accuracy and store it
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy

    return selected_features, accuracies


# A5: Explainability using LIME and SHAP

def explain_with_lime(model, X_train, X_test, feature_names):
    """
    Use LIME to explain a single prediction from the classifier.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=[str(i) for i in np.unique(y)],
        mode="classification"
    )
    
    # Explain the prediction of the first test instance
    i = 0
    exp = explainer.explain_instance(X_test.values[i], model.predict_proba, num_features=7)
    exp.show_in_notebook(show_table=True)
    return exp

def explain_with_shap(model, X_train, X_test):
    """
    Use SHAP to explain model predictions on a test set.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot (global feature importance)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # Detailed feature impact on a single instance
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0], matplotlib=True)
    return shap_values

# A1
df = load_dataset("/content/training_mathbert.xlsx")
corr_matrix = compute_correlation_matrix(df)
plot_correlation_heatmap(corr_matrix)

#A2
df = bin_output_column(df)

# Original Features
X, y = split_features_labels(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
orig_results = train_classifiers(X_train, y_train, X_test, y_test)


# PCA: 99% VARIANCE
X_pca_99, _ = apply_pca(X, variance_threshold=0.99)
X_pca99_train, X_pca99_test, _, _ = train_test_split(X_pca_99, y, test_size=0.2, random_state=42)
pca_99_results = train_classifiers(X_pca99_train, y_train, X_pca99_test, y_test)

#A3
# PCA: 95% VARIANCE
X_pca_95, _ = apply_pca(X, variance_threshold=0.95)
X_pca95_train, X_pca95_test, _, _ = train_test_split(X_pca_95, y, test_size=0.2, random_state=42)
pca_95_results = train_classifiers(X_pca95_train, y_train, X_pca95_test, y_test)

print("\nPCA Results")
print("PCA (99% Variance) Accuracy:")
for model_name, accuracy in pca_99_results.items():
    print(f"{model_name}: {accuracy:.4f}")

print("\nPCA (95% Variance) Accuracy:")
for model_name, accuracy in pca_95_results.items():
    print(f"{model_name}: {accuracy:.4f}")

#A4 - Sequential Feature Selection using top 5 features
selected = apply_sequential_feature_selection(X, y, k_features=5)
sfs_results = {}

for model_name, feat_cols in selected.items():
    # Subset original data to selected features
    X_sfs = X[feat_cols]

    # Train-test split on reduced feature set
    X_train_sfs, X_test_sfs, _, _ = train_test_split(X_sfs, y, test_size=0.2, random_state=42)

    # Initialize appropriate model for evaluation
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(random_state=42)

    # Train and evaluate model on selected features
    model.fit(X_train_sfs, y_train)
    preds = model.predict(X_test_sfs)
    acc = accuracy_score(y_test, preds)

    # Save model accuracy
    sfs_results[model_name] = acc

    # Optional: print which features were selected
    print(f"{model_name} selected features: {feat_cols}")


print(" Sequential Feature Selection Results ")
for model_name, accuracy in sfs_results.items():
    print(f"{model_name} Accuracy with Selected Features: {accuracy:.4f}")

#A5 - LIME and SHAP explaination
X, y = split_features_labels(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model to be explained
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# LIME explanation
lime_exp = explain_with_lime(rf_model, X_train, X_test, X.columns.tolist())

# SHAP explanation
shap_values = explain_with_shap(rf_model, X_train, X_test)

selected_features, accuracies = apply_sequential_feature_selection(X, y, k_features=5)

# Print LIME and SHAP explanation results
print("\n LIME Explanation:")
lime_exp.show_in_notebook(show_table=True)

print("\n SHAP Explanation:")
# SHAP summary plot (global feature importance)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# SHAP force plot for the first test instance
shap.initjs()
shap.force_plot(shap_values[0], X_test.iloc[0])

# Results of Sequential Feature Selection (features and accuracies)
print("\n Sequential Feature Selection:")
print("\nSelected Features:")
for model_name, features in selected_features.items():
    print(f"{model_name}: {features}")

print("\nModel Accuracies (after SFS):")
for model_name, accuracy in accuracies.items():
    print(f"{model_name}: {accuracy:.4f}")
