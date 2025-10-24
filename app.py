import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# -------------------------------
# Lottie loader (safe helper)
# -------------------------------
def load_lottie_url(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Financial Health Assistant",
    page_icon="💰",
    layout="wide"
)

# -------------------------------
# CSS + Background Video + Spinner overlay behavior
# -------------------------------
def add_css(background_video_url: str):
    # This CSS creates a video background (full-screen), an overlay for readability,
    # and a small JS MutationObserver to darken the overlay while a Streamlit spinner is present.
    st.markdown(f"""
        <style>
        /* App container */
        .stApp {{
            position: relative;
            overflow: hidden;
            min-height: 100vh;
            animation: fadeInApp 1.2s ease-in-out forwards;
        }}

        /* Background video (fixed full-screen) */
        #bg-video {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            object-fit: cover;
            z-index: 0;
            opacity: 1;
            pointer-events: none;
            filter: saturate(1) contrast(0.95);
            transform: translateZ(0); /* help with some rendering issues */
        }}

        /* Overlay to improve contrast and readability */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: linear-gradient(180deg, rgba(8,6,23,0.25) 0%, rgba(0,0,0,0.35) 100%);
            z-index: 1;
            transition: background 300ms ease, opacity 300ms ease;
            pointer-events: none;
        }}

        /* When the body has class overlay-darker (toggled by JS when spinner present) */
        body.overlay-darker .stApp::before {{
            background: linear-gradient(180deg, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.6) 100%);
        }}

        /* Ensure main Streamlit content sits above video and overlay */
        .main, .block-container, .stBlock, .stMarkdown, 
        .stNumberInput, .stButton, .stMetric, .stTabs, 
        section[data-testid="stSidebar"], 
        div[data-testid="column"] {{
            position: relative !important;
            z-index: 2 !important;
            visibility: visible !important;
            opacity: 1 !important;
        }}

        /* Input fields visibility */
        input[type="number"], .stNumberInput, .stTextInput {{
            background: rgba(255, 255, 255, 0.15) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            visibility: visible !important;
            opacity: 1 !important;
        }}

        /* Better visibility for all interactive elements */
        .stNumberInput > div > div > input {{
            background: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            visibility: visible !important;
        }}

        /* Make sure columns are visible */
        [data-testid="column"] {{
            background: rgba(0, 0, 0, 0.3) !important;
            padding: 20px !important;
            border-radius: 15px !important;
            backdrop-filter: blur(10px) !important;
        }}

        /* Metric boxes */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            background: rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(8px) !important;
            padding: 8px !important;
            border-radius: 8px !important;
        }}

        /* Headings and text */
        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: #fff !important;
            text-shadow: 0 1px 10px rgba(0,0,0,0.45);
        }}

        /* Buttons styling */
        div.stButton > button {{
            background: linear-gradient(90deg,#6a0dad,#3a0ca3);
            color: white;
            border-radius: 10px;
            padding: 8px 16px;
            border: none;
            box-shadow: 0 6px 20px rgba(58,12,163,0.25);
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(58,12,163,0.35);
        }}

        /* Recommendation box styles */
        .recommendation {{
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 10px;
            backdrop-filter: blur(6px);
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .rec-good {{ border-left: 4px solid #2ecc71; }}
        .rec-bad {{ border-left: 4px solid #e74c3c; }}
        .rec-neutral {{ border-left: 4px solid #3498db; }}

        /* Animations */
        @keyframes fadeInApp {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Spinner color tweak (makes spinner glow purple-ish) */
        .stSpinner > div {{
            border-top-color: #b388ff !important;
            border-right-color: #9b59ff !important;
            border-bottom-color: #8a2be2 !important;
            border-left-color: #7b1fa2 !important;
        }}
        </style>

        <!-- Background video element -->
        <video autoplay muted loop playsinline id="bg-video">
            <source src="{background_video_url}" type="video/mp4">
            <!-- fallback message -->
        </video>

        <script>
        // MutationObserver to detect Streamlit spinner and toggle a darker overlay
        (function() {{
            const observer = new MutationObserver(() => {{
                const spinner = document.querySelector('.stSpinner');
                if (spinner) {{
                    document.body.classList.add('overlay-darker');
                }} else {{
                    document.body.classList.remove('overlay-darker');
                }}
            }});

            observer.observe(document.body, {{ childList: true, subtree: true }});

            // Initial check (in case spinner exists already)
            if (document.querySelector('.stSpinner')) {{
                document.body.classList.add('overlay-darker');
            }}
        }})();
        </script>
    """, unsafe_allow_html=True)

# -------------------------------
# Synthetic data + models (same robust logic as before)
# -------------------------------
def generate_data(n_samples=200):
    np.random.seed(48)
    data = {
        "income": np.random.randint(200000, 1000000, n_samples),
        "side_income": np.random.randint(0, 100000, n_samples),
        "annual_tax": np.random.randint(10000, 50000, n_samples),
        "loan": np.random.randint(20000, 500000, n_samples),
        "investment": np.random.randint(5000, 80000, n_samples),
        "personal_exp": np.random.randint(50000, 250000, n_samples),
        "emergency_exp": np.random.randint(5000, 50000, n_samples),
        "main_exp": np.random.randint(50000, 200000, n_samples)
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

# -------------------------------
# Recommendation logic & assistant
# -------------------------------
def get_recommendations(values, financial_status):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    recs = []
    if loan > (income * 0.5):
        recs.append({"text": "⚠️ High loan burden! Reduce or refinance debt.", "type": "bad"})
    else:
        recs.append({"text": "✅ Loan levels are manageable.", "type": "good"})
    if investment < (income * 0.1):
        recs.append({"text": "📈 Increase investments — currently below 10% of income.", "type": "bad"})
    else:
        recs.append({"text": "✅ Strong investment ratio.", "type": "good"})
    if emergency_exp < (income * 0.05):
        recs.append({"text": "🚨 Emergency fund too low — aim for 5% of income.", "type": "bad"})
    else:
        recs.append({"text": "✅ Emergency fund is solid.", "type": "good"})
    if (personal_exp + main_exp) > (income + side_income) * 0.6:
        recs.append({"text": "💸 Expenses high — try cutting unnecessary costs.", "type": "bad"})
    else:
        recs.append({"text": "✅ Expense ratio is healthy.", "type": "good"})
    if savings < 0:
        recs.append({"text": "🚨 Overspending! You're spending more than you earn.", "type": "bad"})
    else:
        recs.append({"text": "💰 Good job maintaining savings!", "type": "good"})
    recs.append({"text": f"🎯 Financial Status: {financial_status}", "type": "neutral"})
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
    disposable_income = (total_income / 12) - total_monthly_expense
    required_saving = goal_amount / months
    if disposable_income >= required_saving:
        status = "Feasible"
        advice = f"✅ You can reach your goal! Disposable income ₹{disposable_income:,.0f}/mo covers ₹{required_saving:,.0f}/mo."
    else:
        status = "Not Feasible"
        advice = f"⚠️ Disposable ₹{disposable_income:,.0f}/mo, need ₹{required_saving:,.0f}/mo — cut expenses or increase income."
    return {
        "Required Saving per Month": round(required_saving),
        "Disposable Income per Month": round(disposable_income),
        "Status": status,
        "Advice": advice
    }

# -------------------------------
# Main app
# -------------------------------
def main():
    # Use the new background video URL
    background_video_url = "https://motionbgs.com/media/5530/macos-colorful-wave.960x540.mp4"
    add_css(background_video_url)

    # Initialize or load models in session state
    if 'models' not in st.session_state:
        df = generate_data()
        st.session_state.models = train_models(df)

    scaler, clf, reg, kmeans = st.session_state.models

    # Header
    st.markdown("<h1 style='text-align:center; margin-bottom: 0.2rem;'>💰 Financial Health Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: rgba(255,255,255,0.85)'>Enter your <b>yearly</b> financial details (in ₹) below for analysis.</p>", unsafe_allow_html=True)
    st.write("")

    # Inputs
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("📝 Your Financials")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Main Income (₹ per year)", min_value=0, value=300000, step=10000)
            side_income = st.number_input("Side Income (₹ per year)", min_value=0, value=10000, step=5000)
            annual_tax = st.number_input("Annual Tax (₹)", min_value=0, value=20000, step=5000)
            loan = st.number_input("Loan Payments (₹ per year)", min_value=0, value=90000, step=5000)
        with c2:
            investment = st.number_input("Investments (₹ per year)", min_value=0, value=10000, step=5000)
            personal_exp = st.number_input("Personal Expenses (₹ per year)", min_value=0, value=100000, step=5000)
            emergency_exp = st.number_input("Emergency Fund (₹)", min_value=0, value=15000, step=5000)
            main_exp = st.number_input("Household Expenses (₹ per year)", min_value=0, value=80000, step=5000)

    with col2:
        st.header("🎯 Your Goals")
        goal_amount = st.number_input("Enter goal amount (₹)", min_value=0, value=50000, step=5000)
        time_period = st.number_input("Enter time period (months)", min_value=1, value=6, step=1)
        st.write("")
        analyze_button = st.button("🔍 Analyze My Finances")

    st.divider()

    if analyze_button:
        with st.spinner('Analyzing your financials...'):
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
            st.metric("Estimated Yearly Savings", savings_text)

        tab1, tab2, tab3 = st.tabs(["📊 Expense Breakdown", "💡 Recommendations", "🎯 Goal Planner"])

        with tab1:
            st.subheader("Expense & Savings Allocation")
            labels = ["Loan", "Investment", "Personal Exp.", "Emergency Fund", "Household Exp.", "Annual Tax"]
            sizes = [loan, investment, personal_exp, emergency_exp, main_exp, annual_tax]
            if savings > 0:
                labels.append("Savings")
                sizes.append(savings)
            df_pie = pd.DataFrame({"Category": labels, "Amount": sizes})
            fig = px.pie(df_pie, names='Category', values='Amount', hole=0.6)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=14))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Your Personalized Recommendations")
            anim_map = {
                "good": "https://assets9.lottiefiles.com/packages/lf20_4kx2q32n.json",
                "bad": "https://assets9.lottiefiles.com/packages/lf20_s9b6vh6x.json",
                "neutral": "https://assets9.lottiefiles.com/packages/lf20_touohxv0.json"
            }
            for i, rec in enumerate(result["Recommendations"]):
                anim = load_lottie_url(anim_map.get(rec['type']))
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