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
    layout="centered"
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

    # Calculate savings
    df["total_income"] = df["income"] + df["side_income"]
    df["total_spent"] = df["annual_tax"] + df["loan"] + df["investment"] + df["personal_exp"] + df["emergency_exp"] + df["main_exp"]
    df["savings"] = df["total_income"] - df["total_spent"]
    df["savings"] = df["savings"].apply(lambda x: x if x > 0 else np.random.randint(10_000, 5_00_000))

    # Update labels — Risky → Moderate
    df["status"] = np.where(
        (df["loan"] > df["income"] * 0.7) | (df["personal_exp"] > df["income"] * 0.8),
        "Critical",
        np.where(
            (df["investment"] > df["income"] * 0.2) | (df["savings"] > df["income"] * 0.15),
            "Safe",
            "Moderate"
        )
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
def get_recommendations(values, result, savings):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp = values
    recs = []

    if loan > (income * 0.6):
        recs.append("⚠️ High loan burden! Try refinancing or faster repayment.")
    else:
        recs.append("✅ Loan levels look manageable.")

    if savings < (income * 0.1):
        recs.append("💡 Try saving at least 10–15% of your income.")
    elif investment < (income * 0.2):
        recs.append("📈 Increase investments for better long-term growth.")
    else:
        recs.append("✅ Great balance between saving & investing.")

    if (personal_exp + main_exp) > (income + side_income) * 0.7:
        recs.append("💸 Spending too high. Reduce personal or household costs.")
    else:
        recs.append("✅ Expense ratio is under control.")

    if result == "Critical":
        recs.append("🚨 Focus on debt repayment and cut expenses immediately.")
    elif result == "Moderate":
        recs.append("🟠 Finances are okay, but you could strengthen savings.")
    else:
        recs.append("🟢 You’re financially healthy! Keep the momentum.")

    return recs


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
        advice = f"You need an extra ₹{gap:,.0f}/month. Try trimming some expenses."
    
    return {
        "Required Saving per Month": round(required_saving, 2),
        "Disposable Income per Month": round(disposable, 2),
        "Gap": round(max(gap, 0), 2),
        "Status": status,
        "Advice": advice
    }


# ---------------- STYLING ----------------
def add_css():
    st.markdown("""
        <style>
        .stApp { background: transparent !important; }
        #bg-video { position: fixed; right:0; bottom:0; min-width:100%; min-height:100%; z-index:-1; object-fit:cover; }
        .title-text {
            background: rgba(0,0,0,0.6);
            padding: 15px 40px;
            border-radius: 12px;
            font-size: 2.4rem;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        }
        .glass-box {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(12px);
            border-radius: 12px;
            padding: 18px;
            margin: 12px 0;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
            color: white;
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
            box-shadow: 0px 0px 10px rgba(255,120,150,0.8);
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
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <video autoplay muted loop id="bg-video">
            <source src="https://videos.pexels.com/video-files/5485144/5485144-hd_1920_1080_25fps.mp4" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)


# ---------------- MAIN APP ----------------
def main():
    add_css()
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

    total_income = income + side_income
    total_spent = annual_tax + loan + investment + personal_exp + emergency_exp + main_exp
    savings = total_income - total_spent

    st.subheader("💸 Calculated Savings:")
    st.metric(label="Savings Left", value=f"₹{savings:,.0f}")

    if st.button("🔍 Analyze My Finances"):
        values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp]
        user_data = pd.DataFrame([[
            income, side_income, annual_tax, loan, investment,
            personal_exp, emergency_exp, main_exp,
            total_income, total_spent, savings
        ]], columns=[
            "income", "side_income", "annual_tax", "loan", "investment",
            "personal_exp", "emergency_exp", "main_exp",
            "total_income", "total_spent", "savings"
        ])

        scaled = scaler.transform(user_data)
        status = clf.predict(scaled)[0]
        score = reg.predict(scaled)[0]

        st.subheader("📊 Analysis Result")
        st.write(f"📌 **Status:** {status}")
        st.progress(int(score))
        st.markdown(f"<div class='score-box'>✨ Stability Score: {score:.2f}%</div>", unsafe_allow_html=True)

        with st.expander("💡 Recommendations"):
            for rec in get_recommendations(values, status, savings):
                st.markdown(f"<div class='recommendation'>{rec}</div>", unsafe_allow_html=True)

        labels = ["Loan", "Investment", "Personal", "Emergency", "Household"]
        sizes = [loan, investment, personal_exp, emergency_exp, main_exp]
        fig = px.pie(names=labels, values=sizes, hole=0.55, color=labels, color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig, use_container_width=True)

    # Goal planner
    st.subheader("🎯 Goal Saving Planner")
    goal_amount = st.number_input("Enter goal amount (₹)", min_value=0, value=1_00_000, step=10_000)
    months = st.number_input("Enter time period (months)", min_value=1, value=12, step=1)

    if st.button("📌 Plan My Savings"):
        plan = goal_saving_plan(goal_amount, months, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp)
        st.write(f"📌 **Required Saving/Month:** ₹{plan['Required Saving per Month']}")
        st.write(f"💵 **Disposable Income/Month:** ₹{plan['Disposable Income per Month']}")
        st.write(f"📊 **Status:** {plan['Status']}")
        st.markdown(f"<div class='recommendation'>💡 {plan['Advice']}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
