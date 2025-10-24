import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# ---------------- LOTTIE UTILS ----------------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Financial Health Assistant",
    page_icon="💰",
    layout="wide"
)

# ---------------- STYLING (The "Aesthetic" Upgrade) ----------------
def add_css():
    st.markdown("""
        <style>
        /* --- Base --- */
        body {
            background-color: #0E1117;
        }
        .stApp {
            background: linear-gradient(135deg, #0E1117 0%, #243B55 100%);
            color: #FFFFFF;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
        }
        .stMarkdown, .stNumberInput > label, .stSelectbox > label {
            color: #E0E0E0 !important;
        }
        .main-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #FFFFFF;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        }
        .card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        .stNumberInput input {
            background-color: rgba(0, 0, 0, 0.2);
            color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
        }
        .stButton > button {
            background: linear-gradient(90deg, #4CAF50, #81C784);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-size: 1.1rem;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton > button:hover {
            transform: scale(1.03);
            box-shadow: 0px 5px 15px rgba(76, 175, 80, 0.4);
            opacity: 1;
        }
        [data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.1rem;
            color: #A0A0B0;
        }
        [data-testid="stMetricValue"] {
            font-size: 2.2rem;
            font-weight: bold;
            color: #FFFFFF;
        }
        [data-testid="stMetricValue"] p:contains("Critical") { 
            color: #FF6B6B;
        }
        [data-testid="stMetricValue"] p:contains("Safe") {
            color: #4ECDC4;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #A0A0B0;
            font-size: 1.1rem;
            padding: 10px 15px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #FFFFFF;
            border-bottom: 3px solid #4CAF50;
        }
        .recommendation {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            font-size: 1.05rem;
            color: #E0E0E0;
            border-left-width: 6px;
            border-left-style: solid;
        }
        .rec-good { border-left-color: #4CAF50; }
        .rec-bad { border-left-color: #F44336; }
        .rec-neutral { border-left-color: #2196F3; }
        </style>
        """, unsafe_allow_html=True)

# ---------------- DATA GENERATION (Unchanged) ----------------
def generate_data(n_samples=200):
    np.random.seed(42)
    data = {
        "income": np.random.randint(1_00_000, 1_00_00_000, n_samples),
        "side_income": np.random.randint(10_000, 20_00_000, n_samples),
        "annual_tax": np.random.randint(5_000, 20_00_000, n_samples),
        "loan": np.random.randint(50_000, 50_00_000, n_samples),
        "investment": np.random.randint(10_000, 30_00_000, n_samples),
        "personal_exp": np.random.randint(50_000, 50_00_000, n_samples),
        "emergency_exp": np.random.randint(10_000, 10_00_000, n_samples),
        "main_exp": np.random.randint(50_000, 30_00_000, n_samples)
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

# ---------------- MODEL TRAINING (Unchanged) ----------------
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

# ---------------- RECOMMENDATIONS (Updated for Visibility) ----------------
def get_recommendations(values, financial_status):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    recs = []
    if loan > (income * 0.5):
        recs.append({"text": "⚠️ High loan burden! Your loan payments are over 50% of your main income. Prioritize reducing this debt or explore refinancing at a lower interest rate.", "type": "bad"})
    else:
        recs.append({"text": "✅ Loan levels are under control. Your debt-to-income ratio appears healthy.", "type": "good"})

    if investment < (income * 0.1):
        recs.append({"text": "📈 Increase investments. You're investing less than 10% of your income. Aim to increase this for better long-term financial growth and to beat inflation.", "type": "bad"})
    else:
        recs.append({"text": "✅ Good investment ratio. You're investing a healthy portion of your income.", "type": "good"})

    if emergency_exp < (income * 0.05):
        recs.append({"text": "🚨 Build a stronger emergency fund. Your fund is less than 5% of your income. Aim for at least 3-6 months of living expenses to protect against unexpected events.", "type": "bad"})
    else:
        recs.append({"text": "✅ Emergency fund looks sufficient for now. Keep contributing to it.", "type": "good"})

    if (personal_exp + main_exp) > (income + side_income) * 0.6:
        recs.append({"text": "💸 Expenses are high. Your core expenses are over 60% of your total income. Review your 'Personal' and 'Household' spending to find areas to cut back.", "type": "bad"})
    else:
        recs.append({"text": "✅ Expense ratio is healthy. Your spending is well-managed relative to your income.", "type": "good"})

    if savings < 0:
        recs.append({"text": "🚨 You are overspending! Your expenses exceed your income, resulting in negative savings. This is critical. You must reduce spending immediately.", "type": "bad"})
    else:
        recs.append({"text": "💰 Great job saving! You are living within your means. Keep saving consistently each month.", "type": "good"})

    if financial_status == "Safe":
        recs.append({"text": "🎯 Excellent! Your overall financial status is 'Safe'. Maintain this balance between spending, saving, and investing.", "type": "neutral"})
    elif financial_status == "Moderate":
        recs.append({"text": "⚠️ Your finances are 'Moderate'. You're doing okay but can be improved. Focus on the 'bad' recommendations above to become 'Safe'.", "type": "neutral"})
    else:
        recs.append({"text": "🚨 Your status is 'Critical'. This requires immediate attention. Focus on reducing debt and unnecessary spending to regain control.", "type": "neutral"})
    return recs

# ---------------- ANALYSIS (Updated) ----------------
def financial_assistant(values, scaler, clf, reg, kmeans):
    scaled = scaler.transform([values])
    status = clf.predict(scaled)[0]
    score = reg.predict(scaled)[0]
    cluster = kmeans.predict(scaled)[0]
    cluster_map = {0: "Balanced Saver", 1: "High Spender", 2: "Aggressive Investor"}
    group = cluster_map.get(cluster, "Unknown")
    if values[-1] < 0:
        group = "High Spender"
    result = {
        "Financial Status": status,
        "Stability Score": round(score, 2),
        "Group": group,
        "Recommendations": get_recommendations(values, status)
    }
    return result

# ---------------- SAVING PLANNER (Unchanged) ----------------
def goal_saving_plan(goal_amount, months, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp):
    total_income = income + side_income
    total_monthly_expense = (annual_tax + loan + personal_exp + emergency_exp + main_exp) / 12
    total_monthly_income = total_income / 12
    disposable_income = total_monthly_income - total_monthly_expense
    required_saving = goal_amount / months
    if disposable_income >= required_saving:
        status = "Feasible"
        advice = f"You're on track! Your disposable income of ₹{disposable_income:,.0f}/month is enough to save ₹{required_saving:,.0f}/month."
    else:
        status = "Not Feasible"
        advice = f"This goal is a stretch. Your disposable income is ₹{disposable_income:,.0f}/month, but you need to save ₹{required_saving:,.0f}/month. Look at reducing expenses or increasing income."
    return {
        "Required Saving per Month": round(required_saving),
        "Disposable Income per Month": round(disposable_income),
        "Status": status,
        "Advice": advice
    }

# ---------------- MAIN (Completely new layout) ----------------
def main():
    add_css()
    # --- Load Models (in session state to avoid reloading) ---
    if 'models' not in st.session_state:
        df = generate_data()
        st.session_state.models = train_models(df)
    scaler, clf, reg, kmeans = st.session_state.models

    # ------- Add animated header ---------
    lottie_finance = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")
    st.markdown("<h1 class='main-title'>💰 Financial Health Assistant</h1>", unsafe_allow_html=True)
    st_lottie(lottie_finance, speed=1, height=160, key="finance-header")
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #A0A0B0;'>Enter your <b>yearly</b> financial details (in ₹) below to get a complete analysis and personalized recommendations.</p>", unsafe_allow_html=True)

    # --- Input Section ---
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("📝 Your Financials")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Main Income", min_value=0, value=12_00_000, step=10_000)
            side_income = st.number_input("Side Income", min_value=0, value=2_00_000, step=10_000)
            annual_tax = st.number_input("Annual Tax", min_value=0, value=1_50_000, step=10_000)
            loan = st.number_input("Loan Payments", min_value=0, value=4_00_000, step=10_000)
        with c2:
            investment = st.number_input("Investments", min_value=0, value=1_00_000, step=10_000)
            personal_exp = st.number_input("Personal Expenses", min_value=0, value=6_00_000, step=10_000)
            emergency_exp = st.number_input("Emergency Fund", min_value=0, value=80_000, step=10_000)
            main_exp = st.number_input("Household Expenses", min_value=0, value=3_50_000, step=10_000)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.header("🎯 Your Goals")
        goal_amount = st.number_input("Enter goal amount (₹)", min_value=0, value=1_00_000, step=10_000)
        time_period = st.number_input("Enter time period (months)", min_value=1, value=12, step=1)
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze My Finances")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # --- Output Section (only shows after button click) ---
    if analyze_button:
        # Loading animation for analysis
        lottie_loading = load_lottie_url("https://assets2.lottiefiles.com/private_files/lf30_e3pteeho.json")
        with st.spinner('Analyzing your financials...'):
            st_lottie(lottie_loading, speed=1, height=120, key="loading")
            savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)
            values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
            result = financial_assistant(values, scaler, clf, reg, kmeans)
            plan = goal_saving_plan(goal_amount, time_period, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp)

        # --- Display Metrics ---
        st.header("📈 Your Financial Snapshot")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Financial Status", result['Financial Status'])
        with metric_cols[1]:
            st.metric("Stability Score", f"{result['Stability Score']}%")
        with metric_cols[2]:
            st.metric("Primary Group", result['Group'])
        with metric_cols[3]:
            savings_text = f"₹{savings:,.0f}"
            savings_delta = "Above Zero" if savings >= 0 else "Below Zero"
            st.metric("Estimated Yearly Savings", savings_text, delta_color="normal" if savings >= 0 else "inverse")

        # --- Display Tabs for Details ---
        tab1, tab2, tab3 = st.tabs(["📊 Expense Breakdown", "💡 Recommendations", "🎯 Goal Planner"])

        with tab1:
            st.subheader("Expense & Savings Allocation")
            labels = ["Loan", "Investment", "Personal Exp.", "Emergency Fund", "Household Exp.", "Annual Tax"]
            sizes = [loan, investment, personal_exp, emergency_exp, main_exp, annual_tax]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#9B59B6', '#F06543']
            if savings > 0:
                labels.append("Savings")
                sizes.append(savings)
                colors.append('#2ECC71')
            df_pie = pd.DataFrame({"Category": labels, "Amount": sizes})
            fig = px.pie(df_pie, 
                         names='Category', 
                         values='Amount', 
                         hole=0.6, 
                         color_discrete_sequence=colors)
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                pull=[0.05 if cat == "Savings" else 0 for cat in labels],
                marker=dict(line=dict(color='#0E1117', width=3))
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(color="white", size=14)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Your Personalized Recommendations")
            # Animated icons for recommendation types
            anim_map = {
                "good": "https://assets9.lottiefiles.com/packages/lf20_4kx2q32n.json",
                "bad": "https://assets9.lottiefiles.com/packages/lf20_s9b6vh6x.json",
                "neutral": "https://assets9.lottiefiles.com/packages/lf20_touohxv0.json"
            }
            for i, rec in enumerate(result["Recommendations"]):
                anim = load_lottie_url(anim_map[rec['type']])
                cols = st.columns([0.13, 0.87])
                with cols[0]:
                    st_lottie(anim, height=54, key=f"rec_anim_{i}")
                with cols[1]:
                    st.markdown(f"<div class='recommendation rec-{rec['type']}'>{rec['text']}</div>", unsafe_allow_html=True)

        with tab3:
            st.subheader(f"Your Goal: Save ₹{goal_amount:,.0f} in {time_period} months")
            plan_cols = st.columns(3)
            with plan_cols[0]:
                st.metric("Goal Status", plan['Status'])
            with plan_cols[1]:
                st.metric("Required Saving / Month", f"₹{plan['Required Saving per Month']:,.0f}")
            with plan_cols[2]:
                st.metric("Disposable Income / Month", f"₹{plan['Disposable Income per Month']:,.0f}")
            advice_type = "good" if plan['Status'] == 'Feasible' else "bad"
            st.markdown(f"<div class='recommendation rec-{advice_type}'>💡 {plan['Advice']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
