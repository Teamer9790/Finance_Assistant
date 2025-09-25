import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Finance Assistant",  # 🔹 Title for browser tab
    page_icon="💰",                  # 🔹 Icon for browser tab
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

    # Status labeling
    df["status"] = np.where(
        (df["loan"] > df["income"] * 0.7) | (df["personal_exp"] > df["income"] * 0.8),
        "Critical",
        np.where(df["investment"] > df["income"] * 0.2, "Safe", "Risky")
    )

    # Random stability score
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

    # Rule overrides
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


# ---------------- STYLING ----------------
# ---------------- STYLING ----------------
def add_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.pexels.com/photos/11177799/pexels-photo-11177799.jpeg?cs=srgb&dl=pexels-merlin-11177799.jpg&fm=jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: scroll !important;  /* ✅ makes wallpaper move with page */
        }

        h1, h2, h3, p, label {
            color: white !important;
            text-shadow: 1px 1px 2px black; /* 🔹 improves visibility */
        }

        .title-banner {
            text-align: center;
            margin: 30px 0;
        }
        .title-text {
            background: rgba(0, 0, 0, 0.65);
            display: inline-block;
            padding: 15px 40px;
            border-radius: 12px;
            font-size: 2.2rem;
            font-weight: bold;
            color: white;
        }

        .recommendation {
            background: rgba(255, 255, 255, 0.9); /* 🔹 semi-white */
            border-left: 5px solid #4CAF50;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 6px;
            font-size: 0.95rem;
            color: black; /* 🔹 dark text */
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

    st.markdown("<div class='title-banner'><div class='title-text'>💰 Financial Health Assistant</div></div>", unsafe_allow_html=True)
    st.write("Enter your yearly financial details (in ₹):")

    income = st.number_input("Main Income", min_value=0, value=12_00_000, step=10_000)
    side_income = st.number_input("Side Income", min_value=0, value=2_00_000, step=10_000)
    annual_tax = st.number_input("Annual Tax", min_value=0, value=1_50_000, step=10_000)
    loan = st.number_input("Loan Payments", min_value=0, value=4_00_000, step=10_000)
    investment = st.number_input("Investments", min_value=0, value=1_00_000, step=10_000)
    personal_exp = st.number_input("Personal Expenses", min_value=0, value=6_00_000, step=10_000)
    emergency_exp = st.number_input("Emergency Fund", min_value=0, value=80_000, step=10_000)
    main_exp = st.number_input("Household Expenses", min_value=0, value=3_50_000, step=10_000)

    if st.button("Analyze My Finances"):
        values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp]
        result = financial_assistant(values, scaler, clf, reg, kmeans)

        st.subheader("📊 Analysis Result")
        st.success(f"**Status:** {result['Financial Status']}")
        st.info(f"**Stability Score:** {result['Stability Score']}")
        st.warning(f"**Group:** {result['Group']}")

        st.subheader("📈 Expense Breakdown")
        labels = ["Loan", "Investment", "Personal", "Emergency", "Household"]
        sizes = [loan, investment, personal_exp, emergency_exp, main_exp]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.axis("equal")
        st.pyplot(fig)

        st.subheader("💡 Recommendations")
        for rec in result["Recommendations"]:
            st.markdown(f"<div class='recommendation'>{rec}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
