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

    # Avoid negatives for cleaner training
    df["savings"] = df["savings"].apply(lambda x: x if x > 0 else np.random.randint(10_000, 5_00_000))

    # Classification logic
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
    income, loan, investment, personal_exp = values
    recs = []

    if loan > (income * 0.6):
        recs.append("⚠️ Loan burden is too high. Focus on repayment.")
    elif savings < (income * 0.1):
        recs.append("💡 Try saving at least 10–15% of your income.")
    elif investment < (income * 0.2):
        recs.append("📈 Consider increasing your investments for better stability.")
    else:
        recs.append("✅ Your finances look strong!")

    if result == "Critical":
        recs.append("🚨 Prioritize debt reduction and cut down expenses.")
    elif result == "Moderate":
        recs.append("🟠 Keep an eye on spending. You’re doing okay, but could do better.")
    else:
        recs.append("🟢 You’re in a great financial position.")

    return recs


# ---------------- MAIN APP ----------------
st.title("💰 Personal Finance Analyzer")
st.write("Analyze your income, expenses, and financial health in seconds.")

df = generate_data()
scaler, clf, reg, kmeans = train_models(df)

# User inputs
income = st.number_input("Main Income", min_value=10000, step=10000, value=500000)
side_income = st.number_input("Side Income", min_value=0, step=5000, value=50000)
annual_tax = st.number_input("Annual Tax", min_value=0, step=5000, value=30000)
loan = st.number_input("Loan Amount", min_value=0, step=10000, value=150000)
investment = st.number_input("Investments", min_value=0, step=10000, value=80000)
personal_exp = st.number_input("Personal Expenses", min_value=0, step=10000, value=200000)
emergency_exp = st.number_input("Emergency Expenses", min_value=0, step=5000, value=20000)
main_exp = st.number_input("Main Expenses", min_value=0, step=10000, value=100000)

# Calculate savings automatically
total_income = income + side_income
total_spent = annual_tax + loan + investment + personal_exp + emergency_exp + main_exp
savings = total_income - total_spent

st.subheader("💸 Calculated Savings:")
st.metric(label="Savings Left", value=f"₹{savings:,.0f}")

# Prediction
user_data = pd.DataFrame([[
    income, side_income, annual_tax, loan, investment,
    personal_exp, emergency_exp, main_exp,
    total_income, total_spent, savings
]], columns=[
    "income", "side_income", "annual_tax", "loan", "investment",
    "personal_exp", "emergency_exp", "main_exp",
    "total_income", "total_spent", "savings"
])

user_scaled = scaler.transform(user_data)
status = clf.predict(user_scaled)[0]
score = reg.predict(user_scaled)[0]
cluster = kmeans.predict(user_scaled)[0]

st.subheader("📊 Financial Status")
st.write(f"**Status:** {status}")
st.write(f"**Stability Score:** {score:.1f} / 100")
st.write(f"**Cluster Group:** {cluster}")

# Recommendations
st.subheader("🧭 Recommendations")
for rec in get_recommendations((income, loan, investment, personal_exp), status, savings):
    st.write("-", rec)

# Visualization
fig = px.scatter(
    df, x="investment", y="savings", color="status",
    size="income", hover_data=["loan", "personal_exp"]
)
st.plotly_chart(fig, use_container_width=True)
