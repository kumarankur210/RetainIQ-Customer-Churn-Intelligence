# RetainIQ – Customer Churn Intelligence

**An End-to-End Customer Churn Analytics and Prediction Project using Power BI, Python, Machine Learning, and Streamlit.**

RetainIQ is a customer churn intelligence project designed to help businesses understand customer behavior, identify churn patterns, evaluate customer churn risk, and support data-driven retention strategies.

The project combines an interactive **Power BI dashboard** with a **machine learning-powered Streamlit application** to connect business intelligence with predictive analytics.

---

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Understanding why customers leave and identifying customers who may be at risk can help businesses improve retention strategies.

RetainIQ addresses this problem through three stages:

- **Analyze:** Explore customer data and discover churn patterns using Power BI.
- **Predict:** Estimate customer churn probability using a machine learning model.
- **Recommend:** Translate analytical findings into actionable business recommendations.

## Project Components

| Component | Description | Technology |
|---|---|---|
| Business Intelligence Dashboard | Interactive customer churn analysis and KPI reporting | Power BI |
| Machine Learning Application | Customer churn prediction and model evaluation | Python, Scikit-learn |
| Interactive Web Interface | Customer segmentation, risk analysis, and CSV exports | Streamlit |

---

# 1. Power BI – Customer Churn Dashboard

The Power BI dashboard provides an interactive overview of customer churn, retention, contract types, payment methods, internet services, and customer behavior.

### Key Performance Indicators (KPIs)

| KPI | Description |
|---|---|
| Total Customers | Overall number of customers |
| Churned Customers | Number of customers who discontinued the service |
| Churn Rate | Percentage of customers who churned |
| Average Monthly Charges | Average monthly customer charges |

### Dashboard Visualizations

**1. Customer Churn vs Tenure**

Analyzes churn rate across different customer tenure groups to understand how customer retention changes over time.

**2. Churned Customers by Contract Type**

Compares retained and churned customer counts across contract categories.

**3. Customer Distribution by Payment Method**

Visualizes how customers are distributed across different payment methods.

**4. Churn Rate by Internet Service**

Compares churn rates across internet service categories, including Fiber Optic, DSL, and customers without internet service.

**5. Monthly Charges vs Tenure**

Uses a scatter chart to explore the relationship between customer tenure and monthly charges, with customer churn status represented through different series.

### Interactive Filters

The dashboard includes slicers for:

- Contract Type
- Internet Service
- Senior Citizen

These filters allow users to explore customer segments and analyze how different selections affect the dashboard visuals.

---

# 2. Python – Machine Learning Application

RetainIQ includes an interactive Streamlit application with five sections.

### Executive Overview

- Total customers
- Churned customers
- Retained customers
- Overall churn rate
- Churn distribution
- Contract-wise churn analysis
- Customer data preview

### Customer Insights

- Customer segmentation
- Segment-wise customer counts
- Churned customer counts
- Segment-wise churn rates
- Filters for contract, internet service, payment method, and churn status
- Filtered data export to CSV

### Model Performance

Evaluates the saved machine learning model using:

- Accuracy
- ROC-AUC
- Classification report
- Confusion matrix
- ROC curve
- Downloadable evaluation report

### Churn Prediction Lab

Allows users to enter customer information and estimate individual churn probability.

Features include:

- Interactive customer input form
- Churn prediction
- Churn probability estimation
- Customer risk classification

| Risk Category | Churn Probability |
|---|---|
| High Risk | ≥ 60% |
| Medium Risk | ≥ 30% and below 60% |
| Low Risk | Below 30% |

The application can also analyze a precomputed customer risk CSV when available.

*Risk thresholds are demonstration thresholds and should be validated before business deployment.*

### Business Recommendations

Provides retention suggestions based on observed customer churn patterns, including:

- Reviewing month-to-month customer retention.
- Improving early-tenure customer engagement.
- Investigating payment-method-related churn patterns.
- Prioritizing high-risk customers for human review.
- Measuring the effectiveness of retention campaigns.

---

# 3. Machine Learning Approach

RetainIQ uses a Scikit-learn Pipeline with a ColumnTransformer for preprocessing and Logistic Regression for classification.

### Numerical Features

- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

**Preprocessing:**
- Missing-value imputation using median.
- Feature scaling using StandardScaler.

### Categorical Features

- gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod

**Preprocessing:**
- Missing-value imputation using the most frequent value.
- One-hot encoding with `handle_unknown="ignore"`.

### Classification Algorithm

**Logistic Regression**

Model configuration:
- `class_weight="balanced"`
- `random_state=42`
- `solver="lbfgs"`
- `max_iter=200`

The trained model is serialized using Joblib and loaded by the Streamlit application for customer churn predictions.

---

# 4. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Business Intelligence | Microsoft Power BI |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Plotly, Power BI |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Development Tools | VS Code, Jupyter Notebook |

---

# 5. Project Structure

```text
RetainIQ-Customer-Churn-Intelligence/
│
├── app.py
│
├── RETAINIQ.pbix
│
├── Telco-Customer-Churn-Cleaned.csv
│
├── retainiq_churn_model.joblib
│
├── RetainIQ_Customer_Churn_Risk.csv
│
├── requirements.txt
│
├── README.md
│
└── Screenshots/
    ├── powerbi-dashboard.png
    ├── executive-overview.png
    ├── customer-insights.png
    ├── model-performance.png
    └── churn-prediction.png
```

*The customer risk CSV is optional. Update filenames and screenshot entries if your repository uses a different structure.*

---

# 6. Dataset

The project uses the **Telco Customer Churn dataset**.

The dataset contains customer demographics, subscription information, service details, billing information, and churn status.

Important fields include:

- customerID
- gender
- SeniorCitizen
- tenure
- Contract
- InternetService
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn

The cleaned dataset is used by the Streamlit application for customer analysis and prediction.

---

# 7. Run the Streamlit Application Locally

### Step 1: Clone the repository

```bash
git clone https://github.com/kumarankur210/RetainIQ-Customer-Churn-Intelligence.git
```

### Step 2: Navigate to the project directory

```bash
cd RetainIQ-Customer-Churn-Intelligence
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the application

```bash
streamlit run app.py
```

The application will open in your default browser.

### Open the Power BI Dashboard

Open `RETAINIQ.pbix` using Microsoft Power BI Desktop.

---

# 8. Business Value

RetainIQ demonstrates how businesses can combine business intelligence and machine learning to:

- Monitor customer churn and retention KPIs.
- Identify customer segments with higher churn rates.
- Understand customer subscription and payment patterns.
- Estimate individual customer churn risk.
- Support targeted customer retention strategies.
- Convert customer data into actionable business insights.

---

# 9. Limitations

- Model predictions depend on the quality and characteristics of the available dataset.
- Model performance should be validated using genuinely unseen data.
- Risk categories use configurable demonstration thresholds.
- Predictions represent estimated risk, not certainty that a customer will leave.
- Observed associations do not establish causation.
- Business recommendations require validation through controlled experiments and campaign measurement.

---

# 10. Future Improvements

- Deploy the Streamlit application online.
- Add automated data upload and refresh workflows.
- Integrate automated model retraining.
- Add customer-level retention recommendations.
- Monitor model performance and prediction drift.
- Expand dashboard reporting and interactive analytics.

---

# Author

**Ankur Kumar**

B.Tech – Information Technology

GitHub: [kumarankur210](https://github.com/kumarankur210)

Project Repository: [RetainIQ – Customer Churn Intelligence](https://github.com/kumarankur210/RetainIQ-Customer-Churn-Intelligence)

---

**RetainIQ – Turning Customer Data into Retention Insights.**
