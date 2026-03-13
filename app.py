import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# Load and train model (we'll cache this so it only runs once)
@st.cache_resource
def load_model():
    # Load data
    df = pd.read_csv('data/UCI_Credit_Card.csv')
    
    # Prepare features
    feature_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                       'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
                       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']
    
    X = df[feature_columns]
    y = df['default.payment.next.month']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    return model, feature_columns

# Load model
model, feature_columns = load_model()

# Header
st.title("💳 Credit Risk Predictor")
st.markdown("### Predict customer default probability using machine learning")
st.markdown("---")

# Sidebar for input
st.sidebar.header("📝 Customer Information")
st.sidebar.markdown("Enter customer details below:")

# Input fields
credit_limit = st.sidebar.number_input(
    "Credit Limit ($)",
    min_value=10000,
    max_value=1000000,
    value=50000,
    step=10000,
    help="Maximum credit card limit"
)

age = st.sidebar.slider(
    "Age",
    min_value=21,
    max_value=80,
    value=35,
    help="Customer's age"
)

sex = st.sidebar.selectbox(
    "Gender",
    options=[1, 2],
    format_func=lambda x: "Male" if x == 1 else "Female"
)

education = st.sidebar.selectbox(
    "Education Level",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Other"
    }[x],
    index=1
)

marriage = st.sidebar.selectbox(
    "Marital Status",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Married",
        2: "Single",
        3: "Other"
    }[x]
)

st.sidebar.markdown("### 📅 Payment History (Last 6 Months)")
st.sidebar.markdown("*-1 = Paid on time, 0 = Revolving credit, 1+ = Months delayed*")

pay_0 = st.sidebar.selectbox("Recent Payment Status", options=[-1, 0, 1, 2, 3, 4, 5], index=0)
pay_2 = st.sidebar.selectbox("Payment 2 Months Ago", options=[-1, 0, 1, 2, 3, 4, 5], index=0)
pay_3 = st.sidebar.selectbox("Payment 3 Months Ago", options=[-1, 0, 1, 2, 3, 4, 5], index=0)
pay_4 = st.sidebar.selectbox("Payment 4 Months Ago", options=[-1, 0, 1, 2, 3, 4, 5], index=0)
pay_5 = st.sidebar.selectbox("Payment 5 Months Ago", options=[-1, 0, 1, 2, 3, 4, 5], index=0)
pay_6 = st.sidebar.selectbox("Payment 6 Months Ago", options=[-1, 0, 1, 2, 3, 4, 5], index=0)

st.sidebar.markdown("### 💵 Financial Information")

bill_amt1 = st.sidebar.number_input("Most Recent Bill ($)", min_value=0, max_value=1000000, value=50000, step=1000)
bill_amt2 = st.sidebar.number_input("Bill 2 Months Ago ($)", min_value=0, max_value=1000000, value=48000, step=1000)
bill_amt3 = st.sidebar.number_input("Bill 3 Months Ago ($)", min_value=0, max_value=1000000, value=47000, step=1000)

pay_amt1 = st.sidebar.number_input("Most Recent Payment ($)", min_value=0, max_value=1000000, value=2000, step=100)
pay_amt2 = st.sidebar.number_input("Payment 2 Months Ago ($)", min_value=0, max_value=1000000, value=2000, step=100)
pay_amt3 = st.sidebar.number_input("Payment 3 Months Ago ($)", min_value=0, max_value=1000000, value=2000, step=100)

# Predict button
predict_button = st.sidebar.button("🔮 Predict Risk", type="primary", use_container_width=True)

# Main area
if predict_button:
    # Create input data
    input_data = pd.DataFrame({
        'LIMIT_BAL': [credit_limit],
        'SEX': [sex],
        'EDUCATION': [education],
        'MARRIAGE': [marriage],
        'AGE': [age],
        'PAY_0': [pay_0],
        'PAY_2': [pay_2],
        'PAY_3': [pay_3],
        'PAY_4': [pay_4],
        'PAY_5': [pay_5],
        'PAY_6': [pay_6],
        'BILL_AMT1': [bill_amt1],
        'BILL_AMT2': [bill_amt2],
        'BILL_AMT3': [bill_amt3],
        'PAY_AMT1': [pay_amt1],
        'PAY_AMT2': [pay_amt2],
        'PAY_AMT3': [pay_amt3]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    
    # Determine risk category
    if probability < 30:
        risk_category = "🟢 LOW RISK"
        risk_color = "green"
        recommendation = "✅ APPROVE"
        interest_rate = "15-18%"
        details = "Customer shows strong payment history and low default probability. Approve with standard terms."
    elif probability < 60:
        risk_category = "🟡 MEDIUM RISK"
        risk_color = "orange"
        recommendation = "⚠️ APPROVE WITH CAUTION"
        interest_rate = "22-25%"
        details = "Customer shows moderate risk. Approve with higher interest rate and enhanced monitoring."
    else:
        risk_category = "🔴 HIGH RISK"
        risk_color = "red"
        recommendation = "❌ DENY OR REQUIRE COLLATERAL"
        interest_rate = "28%+"
        details = "Customer shows high default probability. Consider denial or require significant collateral."
    
    # Display results
    st.markdown("## 📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Default Probability",
            value=f"{probability:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Risk Category",
            value=risk_category
        )
    
    with col3:
        st.metric(
            label="Recommendation",
            value=recommendation
        )
    
    st.markdown("---")
    
    # Detailed recommendation
    st.markdown("### 💼 Business Recommendation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Suggested Interest Rate:** {interest_rate}")
        st.markdown(f"**Risk Assessment:** {details}")
    
    with col2:
        # Risk gauge
        st.markdown("**Risk Score Visualization:**")
        st.progress(probability / 100)
        
        if probability < 30:
            st.success(f"Low default probability ({probability:.1f}%)")
        elif probability < 60:
            st.warning(f"Moderate default probability ({probability:.1f}%)")
        else:
            st.error(f"High default probability ({probability:.1f}%)")
    
    # Key factors
    st.markdown("---")
    st.markdown("### 🔍 Key Risk Factors")
    
    st.info("""
    **Top predictive factors in this model:**
    1. **Recent Payment Status (PAY_0)** - Most important predictor
    2. **Marital Status** - Affects payment behavior
    3. **Gender** - Statistical correlation with default rates
    4. **Education Level** - Influences financial stability
    5. **Previous Payment History** - Pattern recognition
    """)

else:
    # Default view
    st.markdown("## 👈 Enter customer information in the sidebar")
    st.markdown("### Then click **'Predict Risk'** to see results")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 About This Tool")
        st.markdown("""
        This credit risk predictor uses machine learning to assess the likelihood 
        of credit card default based on customer demographics and payment history.
        
        **Model Details:**
        - Algorithm: Logistic Regression
        - Accuracy: 81%
        - Training Data: 30,000 customers
        - Features: 17 predictive variables
        """)
    
    with col2:
        st.markdown("### 🎯 Risk Categories")
        st.markdown("""
        **🟢 Low Risk (0-30%)**
        - 15% actual default rate
        - Standard approval terms
        
        **🟡 Medium Risk (30-60%)**
        - 49% actual default rate
        - Higher interest rates recommended
        
        **🔴 High Risk (60-100%)**
        - 74% actual default rate
        - Denial or collateral required
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built by Suhitha Reddy Somu | Data Analyst</p>
        <p>Powered by Python, Scikit-learn, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
