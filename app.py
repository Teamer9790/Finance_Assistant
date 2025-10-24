import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# -------------------------------
# Lottie Animation Loader
# -------------------------------
def load_lottie_url(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Financial Health Assistant",
    page_icon="💰",
    layout="wide"
)

# -------------------------------
# Add Custom CSS with Background Video
# -------------------------------
def add_css():
    st.markdown("""
        <style>
        /* Fullscreen video background */
        .stApp {
            position: relative;
            overflow: hidden;
        }

        #bg-video {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            object-fit: cover;
            z-index: 0;
        }

        /* Dark overlay for readability */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.55);
            z-index: 1;
        }

        /* Content above background */
        .main, .block-container {
            position: relative;
            z-index: 2;
        }

        /* White text */
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: white !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #00FFAA !important;
        }

        /* Tabs styling */
        .stTabs [role="tab"] {
            background-color: rgba(255,255,255,0.1);
            border-radius: 10px;
            margin-right: 5px;
            padding: 8px 20px;
        }
        .stTabs [role="tab"]:hover {
            background-color: rgba(255,255,255,0.2);
        }

        /* Recommendation boxes */
        .recommendation {
            padding: 10px 16px;
            border-radius: 12px;
            margin-bottom: 10px;
            font-size: 1rem;
            line-height: 1.5;
        }
        .rec-good { background-color: rgba(46, 204, 113, 0.2); }
        .rec-bad { background-color: rgba(231, 76, 60, 0.2); }
        .rec-neutral { background-color: rgba(52, 152, 219, 0.2); }
        </style>

        <!-- Background video -->
        <video autoplay muted loop id="bg-video">
            <source src="https://cdn.pixabay.com/video/2023/05/12/163277-825844251_large.mp4" type="video/mp4">
        </video>
        """, unsafe_allow_html=True)

# -------------------------------
# Generate Synthetic Financial Data
# -------------------------------
def generate_data(n_samples=200):
    np.random.seed(48)
    data = {
        "income": np.random.randint(200000, 1000000, n_samples),
        "side_income": np.random.randint(0, 100000, n_samples),
        "annual_tax": np.random.randint(10000, 50000, n_samples),
        "loan": np.random.randint(20000, 500000, n_samples),
        "investment": np.random.randint(5000, 80000, n_samples),
        "personal_exp": np.random.randint(50000, 250000, n_samples),
        "emergency_exp": np.random.randint(5000, 50000, n_samples),
        "main_exp": np.random.randint(50000, 200000, n_samples)
    }
    df = pd.DataFrame(data)
    df["savings"] = (
        df["income"] + df["side_income"]
        - (df["annual_tax"] + df["loan"] + df["investment"] +
           df["personal_exp"] + df["emergency_exp"] + df["main_exp"])
    )
    df["status"] = np.where(
        (df["loan"] > df["income"] * 0.7) | (df["personal_exp"] > df["income"] * 0.8),
        "Critical",
        np.where(df["investment"] > df["income"] * 0.2, "Safe", "Moderate")
    )
    df["stability_score"] = np.random.randint(20, 100, n_samples)
    return df

# -------------------------------
# Train ML Models
# -------------------------------
def train_models(df):
    X = df.drop(["status", "stability_score"], axis=1)
    y_class = df["status"]
    y_reg = df["stability_score"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(random_state=42).fit(X_scaled, y_class)
    reg = RandomForestRegressor(random_state=42).fit(X_scaled, y_reg)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    return scaler, clf, reg, kmeans

# -------------------------------
# Recommendations System
# -------------------------------
def get_recommendations(values, financial_status):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    recs = []
    if loan > (income * 0.5):
        recs.append({"text": "⚠️ High loan burden! Reduce or refinance debt.", "type": "bad"})
    else:
        recs.append({"text": "✅ Loan levels are manageable.", "type": "good"})
    if investment < (income * 0.1):
        recs.append({"text": "📈 Increase investments — currently below 10% of income.", "type": "bad"})
    else:
        recs.append({"text": "✅ Strong investment ratio.", "type": "good"})
    if emergency_exp < (income * 0.05):
        recs.append({"text": "🚨 Emergency fund too low — aim for 5% of income.", "type": "bad"})
    else:
        recs.append({"text": "✅ Emergency fund is solid.", "type": "good"})
    if (personal_exp + main_exp) > (income + side_income) * 0.6:
        recs.append({"text": "💸 Expenses high — try cutting unnecessary costs.", "type": "bad"})
    else:
        recs.append({"text": "✅ Expense ratio is healthy.", "type": "good"})
    if savings < 0:
        recs.append({"text": "🚨 Overspending! You're spending more than you earn.", "type": "bad"})
    else:
        recs.append({"text": "💰 Great job maintaining savings!", "type": "good"})
    recs.append({"text": f"🎯 Financial Status: {financial_status}", "type": "neutral"})
    return recs

# -------------------------------
# Financial Assistant Model
# -------------------------------
def financial_assistant(values, scaler, clf, reg, kmeans):
    scaled = scaler.transform([values])
    status = clf.predict(scaled)[0]
    score = reg.predict(scaled)[0]
    cluster = kmeans.predict(scaled)[0]
    cluster_map = {0: "Balanced Saver", 1: "High Spender", 2: "Aggressive Investor"}
    group = cluster_map.get(cluster, "Unknown")
    if values[-1] < 0:
        group = "High Spender"
    return {
        "Financial Status": status,
        "Stability Score": round(score, 2),
        "Group": group,
        "Recommendations": get_recommendations(values, status)
    }

# -------------------------------
# Goal Planner
# -------------------------------
def goal_saving_plan(goal_amount, months, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp):
    total_income = income + side_income
    total_monthly_expense = (annual_tax + loan + personal_exp + emergency_exp + main_exp) / 12
    disposable_income = (total_income / 12) - total_monthly_expense
    required_saving = goal_amount / months
    if disposable_income >= required_saving:
        status = "Feasible"
        advice = f"✅ You can reach your goal! Disposable income ₹{disposable_income:,.0f}/mo covers ₹{required_saving:,.0f}/mo."
    else:
        status = "Not Feasible"
        advice = f"⚠️ Disposable ₹{disposable_income:,.0f}/mo, need ₹{required_saving:,.0f}/mo — cut expenses or increase income."
    return {
        "Required Saving per Month": round(required_saving),
        "Disposable Income per Month": round(disposable_income),
        "Status": status,
        "Advice": advice
    }

# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    add_css()

    if 'models' not in st.session_state:
        df = generate_data()
        st.session_state.models = train_models(df)

    scaler, clf, reg, kmeans = st.session_state.models

    # Header
    st.markdown("<h1 style='text-align:center;'>💰 Financial Health Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Analyze and improve your financial health effortlessly.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.header("🧾 Financial Inputs")
        income = st.number_input("Main Income (₹)", min_value=0, value=300000, step=10000)
        side_income = st.number_input("Side Income (₹)", min_value=0, value=10000, step=10000)
        annual_tax = st.number_input("Annual Tax (₹)", min_value=0, value=20000, step=5000)
        loan = st.number_input("Loan Payments (₹)", min_value=0, value=90000, step=10000)
        investment = st.number_input("Investments (₹)", min_value=0, value=10000, step=5000)
        personal_exp = st.number_input("Personal Expenses (₹)", min_value=0, value=100000, step=10000)
        emergency_exp = st.number_input("Emergency Fund (₹)", min_value=0, value=15000, step=5000)
        main_exp = st.number_input("Household Expenses (₹)", min_value=0, value=80000, step=10000)

    with col2:
        st.header("🎯 Goal Planner")
        goal_amount = st.number_input("Goal Amount (₹)", min_value=0, value=50000, step=10000)
        months = st.number_input("Months to Reach Goal", min_value=1, value=6, step=1)
        analyze = st.button("🔍 Analyze My Finances")

    if analyze:
        savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)
        values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
        result = financial_assistant(values, scaler, clf, reg, kmeans)
        plan = goal_saving_plan(goal_amount, months, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp)

        st.divider()
        st.header("📊 Financial Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", result["Financial Status"])
        c2.metric("Stability Score", f"{result['Stability Score']}%")
        c3.metric("Group", result["Group"])
        c4.metric("Savings", f"₹{savings:,.0f}")

        tab1, tab2, tab3 = st.tabs(["💡 Recommendations", "📈 Expense Breakdown", "🎯 Goal Planner"])

        with tab1:
            for rec in result["Recommendations"]:
                st.markdown(f"<div class='recommendation rec-{rec['type']}'>{rec['text']}</div>", unsafe_allow_html=True)

        with tab2:
            st.subheader("Expense Distribution")
            df_pie = pd.DataFrame({
                "Category": ["Loan", "Investment", "Personal Exp.", "Emergency Fund", "Household Exp.", "Annual Tax", "Savings"],
                "Amount": [loan, investment, personal_exp, emergency_exp, main_exp, annual_tax, max(savings, 0)]
            })
            fig = px.pie(df_pie, names='Category', values='Amount', hole=0.6)
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Goal Feasibility")
            st.metric("Goal Status", plan["Status"])
            st.metric("Required Saving / Month", f"₹{plan['Required Saving per Month']:,.0f}")
            st.metric("Disposable Income / Month", f"₹{plan['Disposable Income per Month']:,.0f}")
            st.markdown(f"<div class='recommendation rec-{'good' if plan['Status']=='Feasible' else 'bad'}'>{plan['Advice']}</div>", unsafe_allow_html=True)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    main()
