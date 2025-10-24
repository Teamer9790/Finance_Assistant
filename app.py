import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

st.set_page_config(
    page_title="Financial Health Assistant",
    page_icon="💰",
    layout="wide"
)

def add_css():
    st.markdown("""
        <style>
        /* ... (CSS remains the same as earlier for styling) ... */
        </style>
        """, unsafe_allow_html=True)

def generate_data(n_samples=200):
    np.random.seed(48)
    data = {
        "income": np.random.randint(2_00_000, 10_00_000, n_samples),
        "side_income": np.random.randint(0, 1_00_000, n_samples),
        "annual_tax": np.random.randint(10_000, 50_000, n_samples),
        "loan": np.random.randint(20_000, 5_00_000, n_samples),
        "investment": np.random.randint(5_000, 80_000, n_samples),
        "personal_exp": np.random.randint(50_000, 2_50_000, n_samples),
        "emergency_exp": np.random.randint(5_000, 50_000, n_samples),
        "main_exp": np.random.randint(50_000, 2_00_000, n_samples)
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

def get_recommendations(values, financial_status):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    recs = []
    if loan > (income * 0.5):
        recs.append({"text": "⚠️ High loan burden! Your loan payments are over 50% of your main income. Prioritize reducing this debt or explore refinancing.", "type": "bad"})
    else:
        recs.append({"text": "✅ Loan levels are under control.", "type": "good"})
    if investment < (income * 0.1):
        recs.append({"text": "📈 Increase investments. Less than 10% of income is going to investment.", "type": "bad"})
    else:
        recs.append({"text": "✅ Good investment ratio.", "type": "good"})
    if emergency_exp < (income * 0.05):
        recs.append({"text": "🚨 Build a stronger emergency fund. It's less than 5% of income.", "type": "bad"})
    else:
        recs.append({"text": "✅ Emergency fund looks sufficient.", "type": "good"})
    if (personal_exp + main_exp) > (income + side_income) * 0.6:
        recs.append({"text": "💸 Expenses are high. Review your spending.", "type": "bad"})
    else:
        recs.append({"text": "✅ Expense ratio is healthy.", "type": "good"})
    if savings < 0:
        recs.append({"text": "🚨 Overspending! Expenses exceed income.", "type": "bad"})
    else:
        recs.append({"text": "💰 Great job saving!", "type": "good"})
    if financial_status == "Safe":
        recs.append({"text": "🎯 Status is 'Safe'. Maintain balance.", "type": "neutral"})
    elif financial_status == "Moderate":
        recs.append({"text": "⚠️ 'Moderate'. Improve following 'bad' tips.", "type": "neutral"})
    else:
        recs.append({"text": "🚨 Status is 'Critical'. Reduce debt urgently.", "type": "neutral"})
    return recs

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

def goal_saving_plan(goal_amount, months, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp):
    total_income = income + side_income
    total_monthly_expense = (annual_tax + loan + personal_exp + emergency_exp + main_exp) / 12
    total_monthly_income = total_income / 12
    disposable_income = total_monthly_income - total_monthly_expense
    required_saving = goal_amount / months
    if disposable_income >= required_saving:
        status = "Feasible"
        advice = f"You're on track! Disposable income ₹{disposable_income:,.0f}/month covers ₹{required_saving:,.0f}/month."
    else:
        status = "Not Feasible"
        advice = f"Disposable income ₹{disposable_income:,.0f}/month, but need ₹{required_saving:,.0f}/month. Cut expenses or boost income."
    return {
        "Required Saving per Month": round(required_saving),
        "Disposable Income per Month": round(disposable_income),
        "Status": status,
        "Advice": advice
    }

def main():
    add_css()
    if 'models' not in st.session_state:
        df = generate_data()
        st.session_state.models = train_models(df)
    scaler, clf, reg, kmeans = st.session_state.models

    lottie_finance = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")
    st.markdown("<h1 class='main-title'>💰 Financial Health Assistant</h1>", unsafe_allow_html=True)
    if lottie_finance:
        st_lottie(lottie_finance, speed=1, height=160, key="finance-header")
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #A0A0B0;'>Enter your <b>yearly</b> financial details (in ₹) below for analysis.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("📝 Your Financials")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Main Income", min_value=0, value=3_00_000, step=10_000)
            side_income = st.number_input("Side Income", min_value=0, value=10_000, step=10_000)
            annual_tax = st.number_input("Annual Tax", min_value=0, value=20_000, step=10_000)
            loan = st.number_input("Loan Payments", min_value=0, value=90_000, step=10_000)
        with c2:
            investment = st.number_input("Investments", min_value=0, value=10_000, step=10_000)
            personal_exp = st.number_input("Personal Expenses", min_value=0, value=1_00_000, step=10_000)
            emergency_exp = st.number_input("Emergency Fund", min_value=0, value=15_000, step=5_000)
            main_exp = st.number_input("Household Expenses", min_value=0, value=80_000, step=10_000)

    with col2:
        st.header("🎯 Your Goals")
        goal_amount = st.number_input("Enter goal amount (₹)", min_value=0, value=50_000, step=10_000)
        time_period = st.number_input("Enter time period (months)", min_value=1, value=6, step=1)
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze My Finances")

    st.divider()

    if analyze_button:
        lottie_loading = load_lottie_url("https://assets2.lottiefiles.com/private_files/lf30_e3pteeho.json")
        with st.spinner('Analyzing your financials...'):
            if lottie_loading:
                st_lottie(lottie_loading, speed=1, height=120, key="loading")
            savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)
            values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
            result = financial_assistant(values, scaler, clf, reg, kmeans)
            plan = goal_saving_plan(goal_amount, time_period, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp)

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
            anim_map = {
                "good": "https://assets9.lottiefiles.com/packages/lf20_4kx2q32n.json",
                "bad": "https://assets9.lottiefiles.com/packages/lf20_s9b6vh6x.json",
                "neutral": "https://assets9.lottiefiles.com/packages/lf20_touohxv0.json"
            }
            for i, rec in enumerate(result["Recommendations"]):
                anim = load_lottie_url(anim_map[rec['type']])
                cols = st.columns([0.14, 0.86])
                with cols[0]:
                    if anim:
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
