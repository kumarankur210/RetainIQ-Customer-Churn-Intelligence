
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="RetainIQ | Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "Telco-Customer-Churn-Cleaned.csv"
MODEL_FILE = BASE_DIR / "retainiq_churn_model.joblib"
RISK_FILE = BASE_DIR / "RetainIQ_Customer_Churn_Risk.csv"

# --------------------------------------------------
# PROFESSIONAL DARK THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #06172E;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] {
        background-color: #102A46;
    }
    [data-testid="stMetric"] {
        background-color: #102A46;
        border: 1px solid #24415E;
        padding: 18px;
        border-radius: 12px;
    }
    h1, h2, h3, p, label {
        color: #FFFFFF;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #24415E;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD PROJECT FILES
# --------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE.name}")
    st.info("Place the cleaned CSV in the same folder as app.py.")
    st.stop()

if not MODEL_FILE.exists():
    st.error(f"Model not found: {MODEL_FILE.name}")
    st.info("Run your churn model training script first.")
    st.stop()

df = load_data(str(DATA_FILE))
model = load_model(str(MODEL_FILE))

required_columns = {"customerID", "Churn"}
if not required_columns.issubset(df.columns):
    st.error("The cleaned dataset must contain customerID and Churn.")
    st.stop()

# Standardize target labels for analysis.
df["Churn"] = df["Churn"].astype(str).str.strip()

# Prepare the model features using the same convention
# as the existing training script: exclude ID and target.
feature_columns = [
    col for col in df.columns
    if col not in ["customerID", "Churn"]
]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("RetainIQ")
st.caption("CUSTOMER RETENTION & CHURN INTELLIGENCE PLATFORM")
st.markdown(
    "Explore customer churn patterns, review model performance, "
    "and estimate churn risk for an example customer."
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Customer Insights",
        "Model Performance",
        "Churn Prediction Lab",
        "Business Recommendations",
    ],
)

st.sidebar.divider()
st.sidebar.caption("RetainIQ | Portfolio Demonstration")
st.sidebar.caption("Predictions are estimates, not guarantees.")

# --------------------------------------------------
# SHARED KPI HELPERS
# --------------------------------------------------
total_customers = len(df)
churned_count = int((df["Churn"].str.lower() == "yes").sum())
retained_count = int((df["Churn"].str.lower() == "no").sum())
churn_rate = (
    churned_count / total_customers
    if total_customers else 0
)

# --------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# --------------------------------------------------
if page == "Executive Overview":
    st.header("Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned_count:,}")
    c3.metric("Retained Customers", f"{retained_count:,}")
    c4.metric("Churn Rate", f"{churn_rate:.2%}")

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Churn Breakdown")
        churn_summary = (
            df["Churn"]
            .str.lower()
            .map({"yes": "Churned", "no": "Retained"})
            .value_counts()
            .rename_axis("Customer Status")
            .reset_index(name="Customers")
        )
        fig = px.pie(
            churn_summary,
            names="Customer Status",
            values="Customers",
            hole=0.55,
            color="Customer Status",
            color_discrete_map={
                "Churned": "#F06464",
                "Retained": "#20C5D7",
            },
        )
        fig.update_layout(
            paper_bgcolor="#102A46",
            plot_bgcolor="#102A46",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Churn by Contract")
        if "Contract" in df.columns:
            contract_summary = (
                df.groupby("Contract", dropna=False)["Churn"]
                .apply(lambda s: (s.str.lower() == "yes").mean() * 100)
                .reset_index(name="Churn Rate (%)")
            )
            fig = px.bar(
                contract_summary,
                x="Contract",
                y="Churn Rate (%)",
                color="Contract",
                text_auto=".1f",
                title="Churn rate by contract type",
            )
            fig.update_layout(
                paper_bgcolor="#102A46",
                plot_bgcolor="#102A46",
                font_color="white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

# --------------------------------------------------
# PAGE 2: CUSTOMER INSIGHTS
# --------------------------------------------------
elif page == "Customer Insights":
    st.header("Customer Insights")
    st.write("Explore churn patterns by customer segment.")

    segment_columns = [
        col for col in
        ["Contract", "Tenure Group", "InternetService", "PaymentMethod"]
        if col in df.columns
    ]

    if not segment_columns:
        st.warning("No supported segment columns were found.")
    else:
        selected_segment = st.selectbox(
            "Choose a customer segment",
            segment_columns,
        )

        summary = (
            df.groupby(selected_segment, dropna=False)
            .agg(
                Customers=("Churn", "size"),
                Churned=("Churn", lambda s: (s.str.lower() == "yes").sum()),
            )
            .reset_index()
        )
        summary["Churn Rate (%)"] = (
            summary["Churned"] / summary["Customers"] * 100
        )

        fig = px.bar(
            summary,
            x=selected_segment,
            y="Churn Rate (%)",
            hover_data=["Customers", "Churned"],
            text_auto=".1f",
            color="Churn Rate (%)",
            color_continuous_scale=["#20C5D7", "#7B61FF", "#F06464"],
            title=f"Churn rate by {selected_segment}",
        )
        fig.update_layout(
            paper_bgcolor="#102A46",
            plot_bgcolor="#102A46",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary, use_container_width=True)

    st.subheader("Filter Customers")
    filters = {}

    for col in ["Contract", "InternetService", "PaymentMethod", "Churn"]:
        if col in df.columns:
            options = sorted(df[col].dropna().astype(str).unique())
            filters[col] = st.multiselect(
                f"Filter by {col}",
                options,
                default=[],
            )

    filtered = df.copy()
    for col, selected in filters.items():
        if selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]

    st.write(f"Matching customers: {len(filtered):,}")
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        "Download filtered customer data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="retainiq_filtered_customers.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance Report")
    st.write(
        "Metrics below are calculated on a held-out test split, "
        "not on the full dataset used to generate demo risk scores."
    )

    if not df["Churn"].str.lower().isin(["yes", "no"]).all():
        st.error("Churn target values must be Yes or No.")
        st.stop()

    X = df[feature_columns].copy()
    y = df["Churn"].str.lower().map({"no": 0, "yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    try:
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # Accommodate models trained with Yes/No labels or 0/1 labels.
        classes = list(getattr(model, "classes_", []))
        if classes and classes[1] not in [1, "1", "Yes", "yes"]:
            positive_index = next(
                (i for i, value in enumerate(classes)
                 if str(value).strip().lower() in ["1", "yes"]),
                None,
            )
            if positive_index is not None:
                probabilities = model.predict_proba(X_test)[:, positive_index]

        # Normalize predicted labels to 0/1.
        pred_labels = pd.Series(predictions).astype(str).str.lower()
        pred_binary = pred_labels.map(
            {"no": 0, "yes": 1, "0": 0, "1": 1}
        )
        if pred_binary.isna().any():
            st.error("Could not interpret the model's predicted labels.")
            st.stop()

        auc = roc_auc_score(y_test, probabilities)
        accuracy = accuracy_score(y_test, pred_binary)

        m1, m2, m3 = st.columns(3)
        m1.metric("Test ROC-AUC", f"{auc:.4f}")
        m2.metric("Test Accuracy", f"{accuracy:.2%}")
        m3.metric("Test Customers", f"{len(y_test):,}")

        st.subheader("Classification Report")
        report = classification_report(
            y_test,
            pred_binary,
            labels=[0, 1],
            target_names=["Retained (No)", "Churned (Yes)"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, pred_binary, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Retained", "Actual Churned"],
            columns=["Predicted Retained", "Predicted Churned"],
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(
            paper_bgcolor="#102A46",
            plot_bgcolor="#102A46",
            font_color="white",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_df = pd.DataFrame(
            {"False Positive Rate": fpr, "True Positive Rate": tpr}
        )
        fig_roc = px.line(
            roc_df,
            x="False Positive Rate",
            y="True Positive Rate",
            title=f"ROC Curve (AUC = {auc:.3f})",
        )
        fig_roc.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash"),
        )
        fig_roc.update_layout(
            paper_bgcolor="#102A46",
            plot_bgcolor="#102A46",
            font_color="white",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        st.download_button(
            "Download classification report",
            data=report_df.to_csv().encode("utf-8"),
            file_name="retainiq_model_report.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(
            "The saved model could not score this test split. "
            "Its expected input columns or preprocessing may differ "
            "from the cleaned dataset."
        )
        st.exception(exc)

# --------------------------------------------------
# PAGE 4: CHURN PREDICTION LAB
# --------------------------------------------------
elif page == "Churn Prediction Lab":
    st.header("Churn Prediction Lab")
    st.write(
        "Enter customer details to estimate churn probability. "
        "This is a model estimate, not a guarantee."
    )

    # Use the saved risk file when available.
    if RISK_FILE.exists():
        risk_df = load_data(str(RISK_FILE))
        st.subheader("Existing Customer Risk Scores")

        if "RiskCategory" in risk_df.columns:
            risk_counts = (
                risk_df["RiskCategory"]
                .value_counts()
                .rename_axis("Risk Category")
                .reset_index(name="Customers")
            )
            fig = px.bar(
                risk_counts,
                x="Risk Category",
                y="Customers",
                color="Risk Category",
                color_discrete_map={
                    "High": "#F06464",
                    "Medium": "#F5B942",
                    "Low": "#20C5D7",
                },
                title="Customers by Risk Category",
            )
            fig.update_layout(
                paper_bgcolor="#102A46",
                plot_bgcolor="#102A46",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        if {"ChurnProbability", "RiskCategory", "customerID"}.issubset(risk_df.columns):
            top_risk = risk_df.sort_values(
                "ChurnProbability", ascending=False
            ).head(20)
            st.subheader("Highest-Risk Customers")
            st.dataframe(top_risk, use_container_width=True)

    st.divider()
    st.subheader("Try a New Customer Profile")

    with st.form("prediction_form"):
        input_values = {}

        for col in feature_columns:
            series = df[col]

            if pd.api.types.is_numeric_dtype(series):
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                default_value = float(numeric_series.median()) if not numeric_series.empty else 0.0

                if col == "SeniorCitizen":
                    input_values[col] = st.selectbox(
                        col, options=[0, 1], index=0
                    )
                elif col == "tenure":
                    input_values[col] = st.number_input(
                        col, min_value=0, max_value=72,
                        value=int(np.clip(round(default_value), 0, 72)),
                    )
                else:
                    low = float(numeric_series.min()) if not numeric_series.empty else 0.0
                    high = float(numeric_series.max()) if not numeric_series.empty else 100.0
                    if low == high:
                        input_values[col] = low
                    else:
                        input_values[col] = st.number_input(
                            col,
                            min_value=low,
                            max_value=high,
                            value=float(np.clip(default_value, low, high)),
                        )
            else:
                options = sorted(series.dropna().astype(str).unique())
                if not options:
                    options = ["Unknown"]
                input_values[col] = st.selectbox(col, options=options)

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        input_df = pd.DataFrame([input_values], columns=feature_columns)

        try:
            probability = float(model.predict_proba(input_df)[0, 1])

            classes = list(getattr(model, "classes_", []))
            if classes and str(classes[1]).strip().lower() not in ["1", "yes"]:
                positive_index = next(
                    (i for i, value in enumerate(classes)
                     if str(value).strip().lower() in ["1", "yes"]),
                    None,
                )
                if positive_index is not None:
                    probability = float(model.predict_proba(input_df)[0, positive_index])

            if probability >= 0.60:
                category = "High"
            elif probability >= 0.30:
                category = "Medium"
            else:
                category = "Low"

            st.metric("Estimated Churn Probability", f"{probability:.1%}")
            st.subheader(f"Risk Category: {category}")

            st.progress(min(max(probability, 0.0), 1.0))
            st.caption(
                "Risk thresholds are demonstration settings. "
                "They should be validated against business costs and test data."
            )

        except Exception as exc:
            st.error(
                "Prediction failed. The saved model may expect different "
                "feature columns, data types, or category values."
            )
            st.exception(exc)

# --------------------------------------------------
# PAGE 5: BUSINESS RECOMMENDATIONS
# --------------------------------------------------
elif page == "Business Recommendations":
    st.header("Business Recommendations")

    st.markdown(
        """
        These are proposed actions based on observed churn associations.
        They are hypotheses to test, not proven causes of churn.

        ### 1. Prioritize month-to-month customers
        Review service experience, renewal incentives, and contract
        flexibility for this high-churn segment.

        ### 2. Improve early-tenure retention
        Consider onboarding check-ins during the first 90 days and
        investigate recurring support issues.

        ### 3. Investigate electronic-check customers
        Explore payment friction and offer convenient alternative
        payment methods without assuming payment method causes churn.

        ### 4. Review high-risk model flags
        Use risk scores to prioritize human review, not to automatically
        deny offers or make consequential decisions about customers.

        ### 5. Measure retention campaign outcomes
        Track retention, campaign cost, and incremental improvement
        using a controlled experiment where feasible.
        """
    )

    st.info(
        "The project uses historical Telco customer data. "
        "Business impact and campaign effectiveness have not been "
        "established by this analysis."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "RetainIQ | Portfolio Project | Historical data and model estimates "
    "are for demonstration purposes."
)
