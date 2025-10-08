import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Finance Assistant",
    page_icon="💰",
    layout="centered"
)
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        return None
# ---------------- DATA GENERATION ----------------
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

# ---------------- MODEL TRAINING ----------------
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

# ---------------- RECOMMENDATIONS ----------------
def get_recommendations(values, result):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    recs = []
    if loan > (income * 0.5):
        recs.append("⚠️ High loan burden! Reduce debt or refinance at lower interest.")
    else:
        recs.append("✅ Loan levels are under control.")
    if investment < (income * 0.1):
        recs.append("📈 Increase investments for long-term financial growth.")
    else:
        recs.append("✅ Good investment ratio.")
    if emergency_exp < (income * 0.05):
        recs.append("🚨 Build a stronger emergency fund (at least 5–10% of income).")
    else:
        recs.append("✅ Emergency fund is sufficient.")
    if (personal_exp + main_exp) > (income + side_income) * 0.6:
        recs.append("💸 Expenses are too high compared to income. Cut unnecessary costs.")
    else:
        recs.append("✅ Expense ratio is healthy.")
    if savings < 0:
        recs.append("🚨 You’re overspending! Try cutting down on expenses.")
    else:
        recs.append("💰 Keep saving consistently each month.")
    if result["Financial Status"] == "Safe":
        recs.append("🎯 Excellent! Maintain balance between spending and saving.")
    elif result["Financial Status"] == "Moderate":
        recs.append("⚠️ Finances are okay but can be improved with better planning.")
    else:
        recs.append("🚨 Critical! Focus on reducing debt and unnecessary spending.")
    return recs

# ---------------- ANALYSIS ----------------
def financial_assistant(values, scaler, clf, reg, kmeans):
    scaled = scaler.transform([values])
    status = clf.predict(scaled)[0]
    score = reg.predict(scaled)[0]
    cluster = kmeans.predict(scaled)[0]
    cluster_map = {0: "Saver", 1: "Spender", 2: "Investor"}
    group = cluster_map.get(cluster, "Unknown")
    if values[-1] < 0:
        group = "Spender"
    result = {
        "Financial Status": status,
        "Stability Score": round(score, 2),
        "Group": group,
        "Recommendations": get_recommendations(values, {"Financial Status": status})
    }
    return result

# ---------------- SAVING PLANNER ----------------
def goal_saving_plan(goal_amount, months, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp):
    total_income = income + side_income
    total_expense = annual_tax + loan + personal_exp + emergency_exp + main_exp
    disposable_income = (total_income - total_expense) / 12
    required_saving = goal_amount / months
    if disposable_income >= required_saving:
        status = "Feasible"
        advice = "You're on track to meet your savings goal! Keep saving regularly."
    else:
        status = "Not Feasible"
        advice = "You need to either reduce your expenses or increase income to meet this goal."
    return {
        "Required Saving per Month": round(required_saving),
        "Disposable Income per Month": round(disposable_income),
        "Status": status,
        "Advice": advice
    }

# ---------------- STYLING ----------------
def add_css():
    st.markdown("""
        <style>
        .stApp {background: transparent !important;}
        .title-text {
            background: rgba(0,0,0,0.6);
            padding: 15px 40px;
            border-radius: 12px;
            font-size: 2.4rem;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-bottom: 25px;
        }
        .score-box {
            background: linear-gradient(90deg, #ff7eb3, #ff758c);
            padding: 12px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
            color: white;
            margin-top: 10px;
        }
        .recommendation {
            background: rgba(255,255,255,0.9);
            border-left: 5px solid #4CAF50;
            padding: 10px 15px;
            margin: 6px 0;
            border-radius: 6px;
            font-size: 0.95rem;
            color: black;
        }
        video#bgvid {
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            object-fit: cover;
        }
        </style>
        <video autoplay muted loop id="bgvid">
            <source src="https://www.pexels.com/download/video/5485144/" type="video/mp4">
        </video>
        """, unsafe_allow_html=True)

# ---------------- MAIN ----------------
def main():
    add_css()
    with st.sidebar:
        st.markdown("## 💡 Financial Tips")
        lottie_saving = load_lottie_url("https://lottie.host/4c9e8e7e-4d3a-4b2a-9f5e-4c6c6c6c6c6c/1f0a.json")
        lottie_invest = load_lottie_url("https://lottie.host/3d2a9e3e-2b1a-4c3a-9f5e-4c6c6c6c6c6c/2a0b.json")
        lottie_budget = load_lottie_url("https://lottie.host/7a9e8e7e-4d3a-4b2a-9f5e-4c6c6c6c6c6c/3c0c.json")

        st_lottie(lottie_saving, height=150, key="saving")
        st.markdown("**Save before you spend.** Automate savings to build discipline.")

        st_lottie(lottie_invest, height=150, key="invest")
        st.markdown("**Invest early.** Compounding works best with time.")

        st_lottie(lottie_budget, height=150, key="budget")
        st.markdown("**Track your expenses.** Awareness is the first step to control.")
        df = generate_data()
        scaler, clf, reg, kmeans = train_models(df)

    st.markdown("<div class='title-text'>💰 Financial Health Assistant</div>", unsafe_allow_html=True)
    st.write("Enter your yearly financial details (in ₹):")

    income = st.number_input("Main Income", min_value=0, value=12_00_000, step=10_000)
    side_income = st.number_input("Side Income", min_value=0, value=2_00_000, step=10_000)
    annual_tax = st.number_input("Annual Tax", min_value=0, value=1_50_000, step=10_000)
    loan = st.number_input("Loan Payments", min_value=0, value=4_00_000, step=10_000)
    investment = st.number_input("Investments", min_value=0, value=1_00_000, step=10_000)
    personal_exp = st.number_input("Personal Expenses", min_value=0, value=6_00_000, step=10_000)
    emergency_exp = st.number_input("Emergency Fund", min_value=0, value=80_000, step=10_000)
    main_exp = st.number_input("Household Expenses", min_value=0, value=3_50_000, step=10_000)

    savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)

    if st.button("🔍 Analyze My Finances"):
        values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
        result = financial_assistant(values, scaler, clf, reg, kmeans)

        st.subheader("📊 Analysis Result")
        st.write(f"📌 **Status:** {result['Financial Status']}")
        st.write(f"👥 **Group (Cluster):** {result['Group']}")
        st.progress(int(result["Stability Score"]))
        st.markdown(f"<div class='score-box'>✨ Stability Score: {result['Stability Score']}%</div>", unsafe_allow_html=True)

        if savings >= 0:
            st.write(f"💰 **Estimated Savings:** ₹{savings:,.0f}")
        else:
            st.write("🚨 No savings — spending exceeds income!")

        st.subheader("📈 Expense Breakdown")
        labels = ["Loan", "Investment", "Personal", "Emergency", "Household"]
        sizes = [loan, investment, personal_exp, emergency_exp, main_exp]
        if savings > 0:
            labels.append("Savings")
            sizes.append(savings)
        fig = px.pie(names=labels, values=sizes, hole=0.55, color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=14))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("💡 Recommendations"):
            for rec in result["Recommendations"]:
                st.markdown(f"<div class='recommendation'>{rec}</div>", unsafe_allow_html=True)

    # -------- Goal Saving Planner --------
    st.subheader("🎯 Goal Saving Planner")
    goal_amount = st.number_input("Enter goal amount (₹)", min_value=0, value=1_00_000, step=10_000)
    time_period = st.number_input("Enter time period (months)", min_value=1, value=12, step=1)

    if st.button("📌 Plan My Savings"):
        plan = goal_saving_plan(goal_amount, time_period, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp)

        st.write(f"📌 *Required Saving/Month:* ₹{plan['Required Saving per Month']}")
        st.write(f"💵 *Disposable Income/Month:* ₹{plan['Disposable Income per Month']}")
        st.write(f"📊 *Status:* {plan['Status']}")
        st.markdown(f"<div class='recommendation'>💡 {plan['Advice']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
