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
# Lottie loader
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
# Page config
# -------------------------------
st.set_page_config(
    page_title="Financial Health Assistant",
    page_icon="💰",
    layout="wide"
)

# -------------------------------
# CSS + Background Video
# -------------------------------
def add_css(background_video_url: str):
    st.markdown(f"""
        <style>
        .stApp {{
            position: relative;
            overflow: hidden;
            min-height: 100vh;
        }}
        #bg-video {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            object-fit: cover;
            z-index: 0;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: linear-gradient(180deg, rgba(20,0,40,0.6) 0%, rgba(0,0,0,0.8) 100%);
            z-index: 1;
        }}
        .main, .block-container, .stMarkdown, .stButton, .stNumberInput {{
            position: relative;
            z-index: 2;
        }}
        h1, h2, h3, p, label, span {{
            color: #fff !important;
            text-shadow: 0 1px 8px rgba(0,0,0,0.5);
        }}
        div.stButton > button {{
            background: linear-gradient(90deg,#6a0dad,#3a0ca3);
            color: white;
            border-radius: 10px;
            padding: 8px 16px;
            border: none;
            box-shadow: 0 6px 20px rgba(58,12,163,0.25);
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(58,12,163,0.35);
        }}
        [data-testid="column"] {{
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }}
        </style>
        <video autoplay muted loop playsinline id="bg-video">
            <source src="{background_video_url}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

# -------------------------------
# Synthetic data + models
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
# Recommendation logic
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
        recs.append({"text": "💰 Good job maintaining savings!", "type": "good"})
    recs.append({"text": f"🎯 Financial Status: {financial_status}", "type": "neutral"})
    return recs

# -------------------------------
# Improved grouping logic
# -------------------------------
def financial_assistant(values, scaler, clf, reg, kmeans):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    scaled = scaler.transform([values])
    status = clf.predict(scaled)[0]
    score = reg.predict(scaled)[0]
    cluster = kmeans.predict(scaled)[0]

    # Smart financial behavior labeling
    if savings < 0:
        group = "High Spender"
    elif investment > (income * 0.25):
        group = "Aggressive Investor"
    elif savings > (income * 0.2):
        group = "Balanced Saver"
    else:
        group = "Moderate Planner"

    return {
        "Financial Status": status,
        "Stability Score": round(score, 2),
        "Group": group,
        "Recommendations": get_recommendations(values, status)
    }

# -------------------------------
# Goal saving plan
# -------------------------------
def goal_saving_plan(goal_amount, months, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp):
    total_income = income + side_income
    total_monthly_expense = (annual_tax + loan + personal_exp + emergency_exp + main_exp) / 12
    disposable_income = (total_income / 12) - total_monthly_expense
    required_saving = goal_amount / months
    if disposable_income >= required_saving:
        status = "Feasible"
        advice = f"✅ You can reach your goal! Disposable ₹{disposable_income:,.0f}/mo covers ₹{required_saving:,.0f}/mo."
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
# Main app
# -------------------------------
def main():
    background_video_url = "https://motionbgs.com/media/5348/abstract-color-lines.960x540.mp4"
    add_css(background_video_url)

    if 'models' not in st.session_state:
        df = generate_data()
        st.session_state.models = train_models(df)

    scaler, clf, reg, kmeans = st.session_state.models

    st.markdown("<h1 style='text-align:center;'>💰 Financial Health Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#ccc'>Enter your yearly financial details (₹)</p>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("📝 Your Financials")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Main Income (₹ per year)", min_value=0, value=300000, step=10000)
            side_income = st.number_input("Side Income (₹ per year)", min_value=0, value=10000, step=5000)
            annual_tax = st.number_input("Annual Tax (₹)", min_value=0, value=20000, step=5000)
            loan = st.number_input("Loan Payments (₹ per year)", min_value=0, value=90000, step=5000)
        with c2:
            investment = st.number_input("Investments (₹ per year)", min_value=0, value=10000, step=5000)
            personal_exp = st.number_input("Personal Expenses (₹ per year)", min_value=0, value=100000, step=5000)
            emergency_exp = st.number_input("Emergency Fund (₹)", min_value=0, value=15000, step=5000)
            main_exp = st.number_input("Household Expenses (₹ per year)", min_value=0, value=80000, step=5000)

    with col2:
        st.header("🎯 Your Goals")
        goal_amount = st.number_input("Goal Amount (₹)", min_value=0, value=50000, step=5000)
        months = st.number_input("Time Period (months)", min_value=1, value=6, step=1)
        analyze = st.button("🔍 Analyze My Finances")

    st.divider()

    if analyze:
        with st.spinner('Analyzing your financials...'):
            savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)
            values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
            result = financial_assistant(values, scaler, clf, reg, kmeans)
            plan = goal_saving_plan(goal_amount, months, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp)

        st.header("📈 Your Financial Snapshot")
        cols = st.columns(4)
        cols[0].metric("Financial Status", result['Financial Status'])
        cols[1].metric("Stability Score", f"{result['Stability Score']}%")
        cols[2].metric("Group", result['Group'])
        cols[3].metric("Estimated Savings", f"₹{savings:,.0f}")

        tab1, tab2, tab3 = st.tabs(["📊 Breakdown", "💡 Recommendations", "🎯 Goal Planner"])

        with tab1:
            labels = ["Loan", "Investment", "Personal Exp.", "Emergency Fund", "Household Exp.", "Annual Tax"]
            sizes = [loan, investment, personal_exp, emergency_exp, main_exp, annual_tax]
            if savings > 0:
                labels.append("Savings")
                sizes.append(savings)
            fig = px.pie(pd.DataFrame({"Category": labels, "Amount": sizes}), names='Category', values='Amount', hole=0.6)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            for i, rec in enumerate(result["Recommendations"]):
                st.markdown(f"<div style='margin-bottom:8px;padding:10px;border-radius:8px;background:rgba(255,255,255,0.05);border-left:4px solid #6a0dad;'>{rec['text']}</div>", unsafe_allow_html=True)

        with tab3:
            st.metric("Goal Status", plan['Status'])
            st.metric("Required Saving / Month", f"₹{plan['Required Saving per Month']:,.0f}")
            st.metric("Disposable Income / Month", f"₹{plan['Disposable Income per Month']:,.0f}")
            st.markdown(f"<div style='background:rgba(255,255,255,0.05);padding:10px;border-radius:8px;'>{plan['Advice']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
