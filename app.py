import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Finance Assistant",
    page_icon="💰",
    layout="wide"
)

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

    df["status"] = np.where(
        (df["loan"] > df["income"] * 0.7) | (df["personal_exp"] > df["income"] * 0.8),
        "Critical",
        np.where(df["investment"] > df["income"] * 0.2, "Safe", "Risky")
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
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp = values
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

    if result["Financial Status"] == "Safe":
        recs.append("🎯 Good financial health! Maintain balance of saving & spending.")
    elif result["Financial Status"] == "Risky":
        recs.append("⚠️ Risky finances. Focus on saving more & controlling spending.")
    else:
        recs.append("🚨 Critical state! Prioritize loan reduction & expense control.")

    return recs


# ---------------- ANALYSIS ----------------
def financial_assistant(values, scaler, clf, reg, kmeans):
    scaled = scaler.transform([values])
    status = clf.predict(scaled)[0]
    score = reg.predict(scaled)[0]
    cluster = kmeans.predict(scaled)[0]
    cluster_map = {0: "Saver", 1: "Spender", 2: "Investor"}
    group = cluster_map.get(cluster, "Unknown")

    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp = values

    if investment >= (income * 0.2):
        group = "Investor"
    elif (personal_exp + main_exp) > (income + side_income) * 0.7:
        group = "Spender"
    else:
        group = "Saver"

    result = {
        "Financial Status": status,
        "Stability Score": round(score, 2),
        "Group": group,
        "Recommendations": get_recommendations(values, {"Financial Status": status})
    }
    return result


# ---------------- GOAL SAVING PLANNER ----------------
def goal_saving_plan(goal_amount, months, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp):
    total_income = (income + side_income) / 12
    monthly_tax = annual_tax / 12
    monthly_expenses = (loan/12) + (personal_exp/12) + (emergency_exp/12) + (main_exp/12)
    
    disposable = total_income - monthly_tax - monthly_expenses
    required_saving = goal_amount / months
    gap = required_saving - disposable
    
    if gap <= 0:
        status = "Feasible ✅"
        advice = f"You can achieve your goal by saving ₹{required_saving:,.0f}/month."
    else:
        status = "Not Feasible 🚨"
        advice = f"Need extra ₹{gap:,.0f}/month. Reduce personal expenses by at least this much."
    
    return {
        "Required Saving per Month": round(required_saving, 2),
        "Disposable Income per Month": round(disposable, 2),
        "Gap": round(max(gap, 0), 2),
        "Status": status,
        "Advice": advice
    }


# ---------------- STYLING ----------------
def add_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
        }
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .result-card {
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 15px;
            margin: 15px 0;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }
        .recommendation {
            background: #f9f9f9;
            border-left: 6px solid #4CAF50;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ---------------- MAIN APP ----------------
def main():
    add_css()
    df = generate_data()
    scaler, clf, reg, kmeans = train_models(df)

    st.title("💰 Financial Health Assistant")
    st.markdown("### Enter your yearly financial details (in ₹):")

    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Main Income", min_value=0, value=12_00_000, step=10_000)
        side_income = st.number_input("Side Income", min_value=0, value=2_00_000, step=10_000)
        annual_tax = st.number_input("Annual Tax", min_value=0, value=1_50_000, step=10_000)
        loan = st.number_input("Loan Payments", min_value=0, value=4_00_000, step=10_000)
    with col2:
        investment = st.number_input("Investments", min_value=0, value=1_00_000, step=10_000)
        personal_exp = st.number_input("Personal Expenses", min_value=0, value=6_00_000, step=10_000)
        emergency_exp = st.number_input("Emergency Fund", min_value=0, value=80_000, step=10_000)
        main_exp = st.number_input("Household Expenses", min_value=0, value=3_50_000, step=10_000)

    if st.button("🔍 Analyze My Finances"):
        values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp]
        result = financial_assistant(values, scaler, clf, reg, kmeans)

        st.markdown("## 📊 Analysis Result")
        st.markdown(f"<div class='result-card'><b>Status:</b> {result['Financial Status']}</div>", unsafe_allow_html=True)
        st.progress(int(result["Stability Score"]))  # progress bar for stability
        st.markdown(f"<div class='result-card'><b>Group:</b> {result['Group']}</div>", unsafe_allow_html=True)

        fig = px.pie(
            names=["Loan", "Investment", "Personal", "Emergency", "Household"],
            values=[loan, investment, personal_exp, emergency_exp, main_exp],
            hole=0.4,
            title="Expense Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("💡 Recommendations"):
            for rec in result["Recommendations"]:
                st.markdown(f"<div class='recommendation'>{rec}</div>", unsafe_allow_html=True)

    st.markdown("## 🎯 Goal Saving Planner")
    goal_amount = st.number_input("Goal Amount (₹)", min_value=0, value=1_00_000, step=10_000)
    time_period = st.number_input("Time Period (months)", min_value=1, value=12, step=1)

    if st.button("📌 Plan My Savings"):
        plan = goal_saving_plan(goal_amount, time_period, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp)
        st.markdown(f"<div class='result-card'><b>Required Saving per Month:</b> ₹{plan['Required Saving per Month']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-card'><b>Disposable Income per Month:</b> ₹{plan['Disposable Income per Month']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-card'><b>Status:</b> {plan['Status']}</div>", unsafe_allow_html=True)
        st.info(plan["Advice"])


if __name__ == "__main__":
    main()
