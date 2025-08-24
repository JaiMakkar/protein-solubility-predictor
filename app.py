# ============================================
# Protein Solubility Predictor (Enhanced UI)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import plotly.express as px

# -------------------------
# Page Config + Custom CSS
# -------------------------
st.set_page_config(page_title="Protein Solubility Predictor", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fc; }
    h1, h2, h3 { color: #2E86C1; }
    .stButton>button {
        color: white;
        background: #2E86C1;
        border-radius: 10px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Protein Solubility Prediction in *E. coli*")
st.write("Upload your dataset, train models, and predict solubility with the best ML model.")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["📂 Data", "⚡ Training", "🔮 Prediction"])

# -------------------------
# Tab 1: Data Upload
# -------------------------
with tab1:
    st.subheader("📂 Upload Training Dataset")
    uploaded_file = st.file_uploader("Upload CSV (must include 'Solubility')", type=["csv"])
    if uploaded_file is None:
        st.warning("⚠️ Please upload a dataset to continue.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write("### Preview of Dataset")
    st.dataframe(df.head(20))

    # Preprocess
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X = df.drop(columns=["Solubility"])
    y = df["Solubility"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    st.write("✅ NaNs filled, data scaled, and SMOTE applied")
    st.write("Class distribution after SMOTE:", dict(Counter(y_train_res)))

# -------------------------
# Tab 2: Training
# -------------------------
with tab2:
    st.subheader("⚡ Model Training with GridSearchCV")

    param_grids = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 300],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            "params": {
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=5000, random_state=42),
            "params": {"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["lbfgs"]}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale"]}
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "p": [1, 2]}
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [100, 300], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
        }
    }

    best_model, best_score, results = None, 0, {}
    progress = st.progress(0)
    total_models, done = len(param_grids), 0

    for name, mp in param_grids.items():
        grid = GridSearchCV(mp["model"], mp["params"], cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train_res, y_train_res)
        y_pred = grid.best_estimator_.predict(X_test_scaled)
        y_prob = grid.best_estimator_.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {"best_params": grid.best_params_, "accuracy": acc, "roc_auc": auc}
        if auc > best_score:
            best_score, best_model = auc, grid.best_estimator_

        done += 1
        progress.progress(done / total_models)

    st.success(f"🏆 Best Model: **{best_model.__class__.__name__}** with ROC-AUC: {best_score:.4f}")

    # Metrics Display
    col1, col2 = st.columns(2)
    col1.metric("Best Accuracy", f"{results[list(results.keys())[0]]['accuracy']:.2f}")
    col2.metric("Best ROC-AUC", f"{best_score:.2f}")

    # Results Table
    st.write("### 📊 Model Comparison")
    df_results = pd.DataFrame(results).T
    st.dataframe(df_results)

    # ROC Curve
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig = px.area(x=fpr, y=tpr, title="ROC Curve",
                  labels=dict(x="False Positive Rate", y="True Positive Rate"),
                  width=500, height=400)
    st.plotly_chart(fig)

# -------------------------
# Tab 3: Prediction
# -------------------------
with tab3:
    st.subheader("🔮 Single Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        mw = st.number_input("Molecular Weight", min_value=10000.0, max_value=100000.0, value=50000.0)
    with col2:
        pI = st.number_input("Isoelectric Point", min_value=10000.0, max_value=100000.0, value=50000.0)
    with col3:
        hydro = st.number_input("Hydrophobicity Index", min_value=10000.0, max_value=100000.0, value=50000.0)

    col4, col5 = st.columns(2)
    with col4:
        aroma = st.number_input("Aromaticity", min_value=10000.0, max_value=100000.0, value=50000.0)
    with col5:
        charge = st.number_input("Charge Density", min_value=10000.0, max_value=100000.0, value=50000.0)

    aa_features = np.repeat(1/20, 20)
    user_data = np.array(list(aa_features) + [mw, pI, hydro, aroma, charge]).reshape(1, -1)
    user_data_scaled = scaler.transform(user_data)

    if st.button("Predict Solubility"):
        prediction = best_model.predict(user_data_scaled)[0]
        result = "✅ Soluble" if prediction == 1 else "❌ Insoluble"
        st.write(f"**Prediction:** {result}")

    # Batch Prediction
    st.subheader("📂 Batch Prediction (Upload New Protein Data)")
    batch_file = st.file_uploader("Upload CSV without 'Solubility' column", type=["csv"], key="batch")
    if batch_file is not None:
        new_data = pd.read_csv(batch_file, encoding="latin1")
        st.write("### Uploaded Data Preview")
        st.dataframe(new_data.head())
        try:
            new_data_scaled = scaler.transform(new_data)
            batch_preds = best_model.predict(new_data_scaled)
            results_df = new_data.copy()
            results_df["Predicted_Solubility"] = ["Soluble" if p == 1 else "Insoluble" for p in batch_preds]
            st.write("### Prediction Results")
            st.dataframe(results_df.head(20))
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions as CSV", csv, "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error during batch prediction: {e}")
