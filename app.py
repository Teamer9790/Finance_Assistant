# elite_finance_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Elite Financial Dashboard",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- CACHING & DATA ----------------
@st.cache_data(show_spinner=False)
def generate_data(n_samples=600):
    np.random.seed(42)
    data = {
        "income": np.random.randint(100_000, 10_000_000, n_samples),
        "side_income": np.random.randint(10_000, 2_000_000, n_samples),
        "annual_tax": np.random.randint(5_000, 2_000_000, n_samples),
        "loan": np.random.randint(50_000, 5_000_000, n_samples),
        "investment": np.random.randint(10_000, 3_000_000, n_samples),
        "personal_exp": np.random.randint(50_000, 5_000_000, n_samples),
        "emergency_exp": np.random.randint(10_000, 1_000_000, n_samples),
        "main_exp": np.random.randint(50_000, 3_000_000, n_samples),
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

@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df.drop(["status", "stability_score"], axis=1)
    y_class = df["status"]
    y_reg = df["stability_score"]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    clf = RandomForestClassifier(random_state=42, n_estimators=250, class_weight="balanced").fit(X_scaled, y_class)
    reg = RandomForestRegressor(random_state=42, n_estimators=250).fit(X_scaled, y_reg)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    return scaler, clf, reg, kmeans

# ---------------- DETERMINISTIC RULE ----------------
def determine_rule_status(income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings):
    if (loan > income * 0.7) or (personal_exp > income * 0.8):
        return "Critical"
    if investment > income * 0.2:
        return "Safe"
    return "Moderate"

# ---------------- RECOMMENDATIONS ----------------
def get_recommendations(values, status):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    recs = []
    if loan > income * 0.5:
        recs.append("⚠️ High loan burden — consider refinancing or prioritizing extra payments.")
    else:
        recs.append("✅ Loan levels manageable. Keep monitoring interest rates.")
    if investment < income * 0.1:
        recs.append("📈 Increase investments (consider SIPs / diversified index funds).")
    else:
        recs.append("✅ Investment exposure looks healthy.")
    if emergency_exp < income * 0.05:
        recs.append("🚨 Emergency fund is low — build to 5–10% of income.")
    else:
        recs.append("✅ Emergency buffer OK.")
    if (personal_exp + main_exp) > (income + side_income) * 0.6:
        recs.append("💸 Expense ratio high — identify discretionary spends to cut.")
    else:
        recs.append("✅ Expense ratio acceptable.")
    if savings < 0:
        recs.append("🔻 You're overspending — immediate expense trimming recommended.")
    else:
        recs.append("💰 Continue saving; automate transfers if possible.")
    if status == "Safe":
        recs.append("🎯 Maintain asset allocation & rebalance annually.")
    elif status == "Moderate":
        recs.append("⚠️ Improve investment rate and reduce variable expenses.")
    else:
        recs.append("🚨 Prioritize debt reduction and liquidity.")
    return recs

# ---------------- UI CSS ----------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@600&display=swap');
        html, body, [class*="css"]  {
            background: radial-gradient(circle at 10% 10%, #071029 0%, #0b2b3a 30%, #08121a 100%);
            color: #e6eef6;
            font-family: 'Inter', sans-serif;
        }
        .top-bar {
            display:flex; align-items:center; gap:18px; padding:18px 8px;
        }
        .brand {
            font-family: 'Poppins', sans-serif;
            font-size:22px;
            font-weight:700;
            letter-spacing:0.6px;
            color: linear-gradient(90deg,#56CCF2,#2F80ED);
        }
        .glass {
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 8px 30px rgba(2,6,23,0.6);
        }
        .status-pill {
            padding:8px 14px; border-radius:999px; color:white; font-weight:600; display:inline-block;
        }
        .safe { background: linear-gradient(90deg,#22c55e,#16a34a); }
        .moderate { background: linear-gradient(90deg,#f59e0b,#d97706); }
        .critical { background: linear-gradient(90deg,#ef4444,#b91c1c); }
        .metric {
            background: rgba(255,255,255,0.02);
            padding:12px; border-radius:12px; text-align:center
        }
        .metric h4 { margin:0; font-size:14px; color:#bcd7ee; font-weight:600; }
        .metric p { margin:6px 0 0 0; font-size:20px; font-weight:700; }
        .recommend { border-left:4px solid rgba(99,102,241,0.9); padding:10px 14px; border-radius:8px; background: rgba(255,255,255,0.02); margin-bottom:8px;}
        .small-muted { color:#9fb7d6; font-size:13px; }
        .preset-btn { margin-right:8px; }
        /* nice animated gauge */
        .gauge-wrap { width:100%; background:rgba(255,255,255,0.03); border-radius:12px; padding:8px;}
        .gauge-fill { height:14px; border-radius:10px; background:linear-gradient(90deg,#56CCF2,#2F80ED); transition: width 0.8s ease; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------- PRESETS ----------------
def apply_preset(name):
    presets = {
        "Safe": {
            "income": 1_000_000,
            "side_income": 100_000,
            "annual_tax": 80_000,
            "loan": 150_000,
            "investment": 250_000,
            "personal_exp": 300_000,
            "emergency_exp": 120_000,
            "main_exp": 150_000
        },
        "Moderate": {
            "income": 1_000_000,
            "side_income": 100_000,
            "annual_tax": 100_000,
            "loan": 300_000,
            "investment": 100_000,
            "personal_exp": 400_000,
            "emergency_exp": 50_000,
            "main_exp": 150_000
        },
        "Critical": {
            "income": 1_000_000,
            "side_income": 50_000,
            "annual_tax": 100_000,
            "loan": 800_000,
            "investment": 50_000,
            "personal_exp": 900_000,
            "emergency_exp": 30_000,
            "main_exp": 200_000
        },
        "Edge": {  # borderline safe/moderate
            "income": 1_000_000,
            "side_income": 50_000,
            "annual_tax": 100_000,
            "loan": 700_00,  # 70k was a typo? keep small edge value - but we want borderline: set loan = 210000
            "investment": 210_000,
            "personal_exp": 780_000,
            "emergency_exp": 60_000,
            "main_exp": 140_000
        }
    }
    chosen = presets.get(name)
    if chosen:
        for k, v in chosen.items():
            st.session_state[k] = int(v)
        # recompute savings too (savings computed from inputs on render)
        st.experimental_rerun()

# ---------------- CORE ANALYSIS ----------------
def analyze_and_render(values, scaler, clf, reg, kmeans, show_model_vs_rule=True):
    scaled = scaler.transform([values])
    status_pred = clf.predict(scaled)[0]
    probs = None
    try:
        probs = dict(zip(clf.classes_, (clf.predict_proba(scaled)[0]).round(3)))
    except Exception:
        probs = None
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    status_rule = determine_rule_status(income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings)
    # prefer rule (deterministic) if mismatch to keep consistency
    status = status_rule if status_pred != status_rule else status_pred
    raw_score = reg.predict(scaled)[0]
    score = float(np.clip(round(raw_score, 2), 0.0, 100.0))
    cluster = kmeans.predict(scaled)[0]
    cluster_map = {0: "Saver", 1: "Spender", 2: "Investor"}
    group = cluster_map.get(cluster, "Balanced")
    if savings < 0:
        group = "Spender"
    recs = get_recommendations(values, status)
    return {
        "status": status,
        "status_rule": status_rule,
        "status_pred": status_pred,
        "probs": probs,
        "score": score,
        "group": group,
        "recs": recs,
        "savings": savings
    }

# ---------------- BUILD UI ----------------
def main():
    inject_css()
    df = generate_data()
    scaler, clf, reg, kmeans = train_models(df)

    # initialize session defaults
    defaults = {
        "income": 1_200_000, "side_income": 200_000, "annual_tax": 150_000,
        "loan": 400_000, "investment": 100_000, "personal_exp": 600_000,
        "emergency_exp": 80_000, "main_exp": 350_000
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Top header
    col1, col2 = st.columns([0.7, 1])
    with col1:
        st.markdown("<div class='top-bar'><div style='font-size:26px'>💠</div><div style='margin-left:8px'><div style='font-weight:700; font-size:20px'>Elite Financial Health</div><div class='small-muted'>Premium analyzer • deterministic fallback • transparent</div></div></div>", unsafe_allow_html=True)
    with col2:
        # little quick metrics across top
        st.write("")  # spacer
        metric_cols = st.columns(3)
        for mcol, label in zip(metric_cols, ["Stability", "Estimated Savings", "Group"]):
            with mcol:
                if label == "Stability":
                    st.markdown("<div class='metric'><h4>Stability</h4><p>—</p></div>", unsafe_allow_html=True)
                elif label == "Estimated Savings":
                    st.markdown("<div class='metric'><h4>Estimated Savings</h4><p>—</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='metric'><h4>Group</h4><p>—</p></div>", unsafe_allow_html=True)

    st.markdown("<br>")

    # Main layout: inputs left, analytics right
    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Inputs & Presets")
        p1, p2, p3, p4 = st.columns([1,1,1,1])
        with p1:
            if st.button("Preset: Safe", key="preset_safe"):
                apply_preset("Safe")
        with p2:
            if st.button("Preset: Moderate", key="preset_mod"):
                apply_preset("Moderate")
        with p3:
            if st.button("Preset: Critical", key="preset_crit"):
                apply_preset("Critical")
        with p4:
            if st.button("Preset: Edge", key="preset_edge"):
                apply_preset("Edge")

        st.markdown("<hr>", unsafe_allow_html=True)

        income = st.number_input("💼 Main Income (₹)", min_value=0, value=int(st.session_state.get("income", 1_200_000)), step=10_000, key="income")
        side_income = st.number_input("💸 Side Income (₹)", min_value=0, value=int(st.session_state.get("side_income", 200_000)), step=10_000, key="side_income")
        annual_tax = st.number_input("🧾 Annual Tax (₹)", min_value=0, value=int(st.session_state.get("annual_tax", 150_000)), step=10_000, key="annual_tax")
        loan = st.number_input("🏦 Loan Payments (₹)", min_value=0, value=int(st.session_state.get("loan", 400_000)), step=10_000, key="loan")
        investment = st.number_input("📊 Investments (₹)", min_value=0, value=int(st.session_state.get("investment", 100_000)), step=10_000, key="investment")
        personal_exp = st.number_input("🛍️ Personal Expenses (₹)", min_value=0, value=int(st.session_state.get("personal_exp", 600_000)), step=10_000, key="personal_exp")
        emergency_exp = st.number_input("🚨 Emergency Fund (₹)", min_value=0, value=int(st.session_state.get("emergency_exp", 80_000)), step=10_000, key="emergency_exp")
        main_exp = st.number_input("🏠 Household Expenses (₹)", min_value=0, value=int(st.session_state.get("main_exp", 350_000)), step=10_000, key="main_exp")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Advanced options</div>", unsafe_allow_html=True)
        opts = st.expander("Model options & transparency (expand)")
        with opts:
            st.checkbox("Show Model vs Rule details", value=True, key="show_model_vs_rule")
            st.info("Model is trained on synthetic data. Deterministic rule is used as authoritative fallback for consistency.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        # compute savings
        savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)
        values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
        analysis = analyze_and_render(values, scaler, clf, reg, kmeans, show_model_vs_rule=st.session_state.get("show_model_vs_rule", True))

        # Header row with status and metrics
        hcol1, hcol2, hcol3 = st.columns([1.6, 1, 1])
        with hcol1:
            pill_class = "safe" if analysis["status"] == "Safe" else ("moderate" if analysis["status"] == "Moderate" else "critical")
            st.markdown(f"<div style='display:flex; gap:12px; align-items:center'><div style='font-size:20px; font-weight:700'>Result</div><div class='status-pill {pill_class}'>{analysis['status']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'>Rule used: <strong>{analysis['status_rule']}</strong> • Model predicted: <strong>{analysis['status_pred']}</strong></div>", unsafe_allow_html=True)
        with hcol2:
            st.metric(label="Stability Score", value=f"{analysis['score']}%", delta=None)
        with hcol3:
            s_text = f"₹{analysis['savings']:,.0f}" if analysis['savings'] >= 0 else "− ₹{:,}".format(abs(int(analysis['savings'])))
            st.metric(label="Estimated Savings", value=s_text)

        st.markdown("---")

        # charts and group
        chart_col, info_col = st.columns([1.6, 0.9])
        with chart_col:
            # pie/donut
            labels = ["Loan", "Investment", "Personal", "Emergency", "Household"]
            sizes = [loan, investment, personal_exp, emergency_exp, main_exp]
            if savings > 0:
                labels.append("Savings")
                sizes.append(savings)
            pie = px.pie(names=labels, values=sizes, hole=0.56)
            pie.update_traces(textinfo="percent+label", hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})")
            pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(pie, use_container_width=True)

            # waterfall-like contribution (income vs subtractions)
            contrib_labels = ["Income", "Side Income", "Tax", "Loan", "Investment", "Personal", "Emergency", "Household", "Savings"]
            contrib_values = [income, side_income, -annual_tax, -loan, -investment, -personal_exp, -emergency_exp, -main_exp, savings]
            fig = go.Figure(go.Waterfall(
                x=contrib_labels,
                y=contrib_values,
                measure=["relative","relative","relative","relative","relative","relative","relative","relative","total"],
                text=[f"₹{v:,.0f}" for v in contrib_values],
                connector={"line":{"color":"rgba(63,63,63,0.6)"}}
            ))
            fig.update_layout(title_text="Income & Expense Contribution", showlegend=False, waterfallgap=0.5,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        with info_col:
            st.markdown("### Overview")
            st.markdown(f"- **Group:** {analysis['group']}")
            st.markdown(f"- **Stability:** `{analysis['score']}%`")
            st.markdown(f"- **Savings:** {s_text}")
            st.markdown("---")
            if st.session_state.get("show_model_vs_rule", True):
                st.markdown("### Model vs Rule")
                st.markdown(f"- Deterministic rule result: **{analysis['status_rule']}**")
                st.markdown(f"- Model predicted: **{analysis['status_pred']}**")
                if analysis["probs"]:
                    st.markdown("**Model confidence:**")
                    for cls, p in analysis["probs"].items():
                        st.markdown(f"- {cls}: {p*100:.1f}%")
            st.markdown("---")
            st.markdown("### Recommendations")
            for r in analysis["recs"]:
                st.markdown(f"<div class='recommend'>{r}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # footer quick help
    st.markdown("<div style='padding:12px 6px; text-align:center; color:#9fb7d6'>Tip: Use presets to test Safe / Moderate / Critical. The deterministic rule is authoritative when model disagrees.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
