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

    df["savings"] = (
        df["income"] + df["side_income"]
        - (df["annual_tax"] + df["loan"] + df["investment"] + df["personal_exp"] + df["emergency_exp"] + df["main_exp"])
    )

    # same deterministic rule used both for labeling and as fallback later
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

    # make classifier more robust: balanced class weights, more estimators
    clf = RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced").fit(X_scaled, y_class)
    reg = RandomForestRegressor(random_state=42, n_estimators=200).fit(X_scaled, y_reg)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)

    return scaler, clf, reg, kmeans


# ---------------- DETERMINISTIC RULE (FALLBACK) ----------------
def determine_rule_status(income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings):
    # Use the same rules used in generate_data to ensure consistency
    if (loan > income * 0.7) or (personal_exp > income * 0.8):
        return "Critical"
    if investment > income * 0.2:
        return "Safe"
    return "Moderate"


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
    # classifier prediction
    status_pred = clf.predict(scaled)[0]

    # deterministic rule
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    status_rule = determine_rule_status(income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings)

    # prefer rule-based label if there's a disagreement (keeps consistency with labels used during training)
    if status_pred != status_rule:
        status = status_rule
    else:
        status = status_pred

    # regressor (clamp to 0-100 for percentage-like score)
    raw_score = reg.predict(scaled)[0]
    score = float(np.clip(round(raw_score, 2), 0.0, 100.0))

    cluster = kmeans.predict(scaled)[0]
    cluster_map = {0: "Saver", 1: "Spender", 2: "Investor"}
    group = cluster_map.get(cluster, "Unknown")

    # Override group to Spender if savings are negative (critical spending)
    if savings < 0:
        group = "Spender"

    result = {
        "Financial Status": status,
        "Stability Score": score,
        "Group": group,
        "Recommendations": get_recommendations(values, {"Financial Status": status})
    }
    return result


# ---------------- STYLING ----------------
def add_css():
    st.markdown(
        """
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
        .glass-box {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(12px);
            border-radius: 12px;
            padding: 18px;
            margin: 12px 0;
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
        """,
        unsafe_allow_html=True
    )


# ---------------- MAIN ----------------
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

    savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)

    if st.button("🔍 Analyze My Finances"):
        values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
        result = financial_assistant(values, scaler, clf, reg, kmeans)

        st.subheader("📊 Analysis Result")
        st.write(f"📌 **Status:** {result['Financial Status']}")
        st.write(f"👥 **Group (Cluster):** {result['Group']}")
        # show a bounded progress bar (0-100)
        st.progress(int(np.clip(result["Stability Score"], 0, 100)))
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

        fig = px.pie(
            names=labels,
            values=sizes,
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=14),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("💡 Recommendations"):
            for rec in result["Recommendations"]:
                st.markdown(f"<div class='recommendation'>{rec}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
