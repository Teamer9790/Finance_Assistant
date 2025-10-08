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
    df["savings"] = (
        df["income"] + df["side_income"]
        - (df["annual_tax"] + df["loan"] + df["investment"] + df["personal_exp"] + df["emergency_exp"] + df["main_exp"])
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

    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    if savings < 0:
        group = "Spender"

    result = {
        "Financial Status": status,
        "Stability Score": round(score, 2),
        "Group": group,
        "Recommendations": get_recommendations(values, {"Financial Status": status})
    }
    return result

# ---------------- STYLING ----------------
def add_css():
    st.markdown(
        """
        <style>
        /* Background and layout */
        .stApp {
            background: linear-gradient(135deg, #1c1f2b, #2c3240, #0d1117);
            color: white;
        }

        .main {
            padding: 1rem 3rem;
        }
        /* Fix random bar under title */
        div[data-testid="stVerticalBlock"]:has(> div.title-text) + div:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
}


        /* Title Section */
        .title-text {
            text-align: center;
            font-size: 2.8rem;
            font-weight: 800;
            color: white;
            text-shadow: 0px 0px 12px rgba(255,255,255,0.3);
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #ff8a00, #e52e71);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Glass form card */
        .glass-card {
            background: rgba(255,255,255,0.08);
            padding: 2rem;
            border-radius: 18px;
            backdrop-filter: blur(14px);
            box-shadow: 0 0 15px rgba(255,255,255,0.05);
            margin-bottom: 2rem;
        }

        /* Score box */
        .score-box {
            background: linear-gradient(90deg, #4CAF50, #81C784);
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.2rem;
            font-weight: bold;
            text-align: center;
            color: white;
            margin-top: 1rem;
        }

        /* Recommendation cards */
        .recommendation {
            background: rgba(255,255,255,0.12);
            border-left: 5px solid #4CAF50;
            padding: 0.8rem 1.2rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            color: white;
            font-size: 0.95rem;
        }

        /* Buttons */
        button[data-testid="baseButton-secondary"] {
            background: linear-gradient(90deg, #ff8a00, #e52e71) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease;
        }
        button[data-testid="baseButton-secondary"]:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(255,138,0,0.4);
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

    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📥 Enter Your Yearly Financial Details (₹)")
        income = st.number_input("Main Income", min_value=0, value=12_00_000, step=10_000)
        side_income = st.number_input("Side Income", min_value=0, value=2_00_000, step=10_000)
        annual_tax = st.number_input("Annual Tax", min_value=0, value=1_50_000, step=10_000)
        loan = st.number_input("Loan Payments", min_value=0, value=4_00_000, step=10_000)
        investment = st.number_input("Investments", min_value=0, value=1_00_000, step=10_000)
        personal_exp = st.number_input("Personal Expenses", min_value=0, value=6_00_000, step=10_000)
        emergency_exp = st.number_input("Emergency Fund", min_value=0, value=80_000, step=10_000)
        main_exp = st.number_input("Household Expenses", min_value=0, value=3_50_000, step=10_000)
        st.markdown("</div>", unsafe_allow_html=True)

    savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)

    if st.button("🔍 Analyze My Finances"):
        values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
        result = financial_assistant(values, scaler, clf, reg, kmeans)

        st.markdown("---")
        st.subheader("📊 Analysis Summary")
        st.write(f"📌 **Status:** {result['Financial Status']}")
        st.write(f"👥 **Financial Group:** {result['Group']}")
        st.progress(int(result["Stability Score"]))
        st.markdown(f"<div class='score-box'>✨ Stability Score: {result['Stability Score']}%</div>", unsafe_allow_html=True)

        if savings >= 0:
            st.write(f"💰 **Estimated Savings:** ₹{savings:,.0f}")
        else:
            st.write("🚨 **No Savings — Spending exceeds income!**")

        st.subheader("📈 Expense Breakdown")
        labels = ["Loan", "Investment", "Personal", "Emergency", "Household"]
        sizes = [loan, investment, personal_exp, emergency_exp, main_exp]
        if savings > 0:
            labels.append("Savings")
            sizes.append(savings)

        fig = px.pie(
            names=labels,
            values=sizes,
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=14),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("💡 Personalized Recommendations", expanded=True):
            for rec in result["Recommendations"]:
                st.markdown(f"<div class='recommendation'>{rec}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
