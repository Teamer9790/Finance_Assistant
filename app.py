import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Financial Health Dashboard 💰",
    page_icon="💹",
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

    clf = RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced").fit(X_scaled, y_class)
    reg = RandomForestRegressor(random_state=42, n_estimators=200).fit(X_scaled, y_reg)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)

    return scaler, clf, reg, kmeans


# ---------------- RULES & RECOMMENDATIONS ----------------
def determine_rule_status(income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings):
    if (loan > income * 0.7) or (personal_exp > income * 0.8):
        return "Critical"
    if investment > income * 0.2:
        return "Safe"
    return "Moderate"


def get_recommendations(values, result):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    recs = []

    if loan > (income * 0.5):
        recs.append("⚠️ High loan burden! Reduce or refinance.")
    else:
        recs.append("✅ Loan levels are manageable.")

    if investment < (income * 0.1):
        recs.append("📈 Increase investments for long-term growth.")
    else:
        recs.append("✅ Investment ratio looks solid.")

    if emergency_exp < (income * 0.05):
        recs.append("🚨 Build an emergency fund of at least 5–10% of income.")
    else:
        recs.append("✅ Emergency fund is sufficient.")

    if (personal_exp + main_exp) > (income + side_income) * 0.6:
        recs.append("💸 Expenses are too high! Reduce non-essential spending.")
    else:
        recs.append("✅ Expenses are under control.")

    if savings < 0:
        recs.append("🚨 Overspending detected! Try to reduce expenses.")
    else:
        recs.append("💰 Good savings — keep it consistent.")

    if result["Financial Status"] == "Safe":
        recs.append("🎯 Excellent! Maintain your financial discipline.")
    elif result["Financial Status"] == "Moderate":
        recs.append("⚠️ You’re okay — but focus on improving investments.")
    else:
        recs.append("🚨 Critical! Reduce debts and rebalance your spending.")

    return recs


# ---------------- ANALYSIS ----------------
def financial_assistant(values, scaler, clf, reg, kmeans):
    scaled = scaler.transform([values])
    status_pred = clf.predict(scaled)[0]
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    status_rule = determine_rule_status(income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings)
    status = status_rule if status_pred != status_rule else status_pred
    score = float(np.clip(round(reg.predict(scaled)[0], 2), 0, 100))
    cluster = kmeans.predict(scaled)[0]
    cluster_map = {0: "Saver", 1: "Spender", 2: "Investor"}
    group = cluster_map.get(cluster, "Unknown")
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
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Inter:wght@400;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
            color: white;
        }
        .main-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
            padding: 40px;
            backdrop-filter: blur(15px);
            width: 90%;
            margin: auto;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 12px;
            font-weight: bold;
            margin-top: 10px;
        }
        .Safe {background: linear-gradient(90deg, #00c851, #007e33);}
        .Moderate {background: linear-gradient(90deg, #ffbb33, #ff8800);}
        .Critical {background: linear-gradient(90deg, #ff4444, #cc0000);}
        .score-bar {
            height: 18px;
            border-radius: 12px;
            background: linear-gradient(90deg, #56CCF2, #2F80ED);
            margin-top: 5px;
        }
        .recommend-card {
            background: rgba(255,255,255,0.12);
            border-left: 4px solid #00e676;
            padding: 10px 14px;
            margin: 6px 0;
            border-radius: 8px;
            transition: 0.3s;
        }
        .recommend-card:hover {background: rgba(255,255,255,0.2);}
        .title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(90deg, #56CCF2, #2F80ED);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)


# ---------------- MAIN ----------------
def main():
    add_css()
    df = generate_data()
    scaler, clf, reg, kmeans = train_models(df)

    st.markdown("<div class='title'>💹 Financial Health Dashboard</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)

        income = st.number_input("💼 Main Income", min_value=0, value=12_00_000, step=10_000)
        side_income = st.number_input("💸 Side Income", min_value=0, value=2_00_000, step=10_000)
        annual_tax = st.number_input("🧾 Annual Tax", min_value=0, value=1_50_000, step=10_000)
        loan = st.number_input("🏦 Loan Payments", min_value=0, value=4_00_000, step=10_000)
        investment = st.number_input("📊 Investments", min_value=0, value=1_00_000, step=10_000)
        personal_exp = st.number_input("🛍️ Personal Expenses", min_value=0, value=6_00_000, step=10_000)
        emergency_exp = st.number_input("🚨 Emergency Fund", min_value=0, value=80_000, step=10_000)
        main_exp = st.number_input("🏠 Household Expenses", min_value=0, value=3_50_000, step=10_000)

        savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)

        if st.button("✨ Analyze My Finances", use_container_width=True):
            values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
            result = financial_assistant(values, scaler, clf, reg, kmeans)

            st.markdown(f"<div class='status-badge {result['Financial Status']}'>{result['Financial Status']}</div>", unsafe_allow_html=True)
            st.write(f"👥 **Group:** {result['Group']}")
            st.write(f"💰 **Estimated Savings:** ₹{savings:,.0f}" if savings >= 0 else "🚨 No savings — spending exceeds income!")

            st.progress(int(result["Stability Score"]))
            st.markdown(f"<div class='score-bar' style='width:{result['Stability Score']}%'></div>", unsafe_allow_html=True)
            st.write(f"✨ **Stability Score:** {result['Stability Score']}%")

            labels = ["Loan", "Investment", "Personal", "Emergency", "Household"]
            sizes = [loan, investment, personal_exp, emergency_exp, main_exp]
            if savings > 0:
                labels.append("Savings")
                sizes.append(savings)

            fig = px.pie(names=labels, values=sizes, hole=0.55,
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", size=14),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💡 Recommendations")
            for rec in result["Recommendations"]:
                st.markdown(f"<div class='recommend-card'>{rec}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
