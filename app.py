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
    st.markdown(f"""
        <style>
        .stApp {{
            position: relative;
            overflow: hidden;
            min-height: 100vh;
            animation: fadeInApp 1.2s ease-in-out forwards;
        }}

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
            transform: translateZ(0);
        }}

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

        body.overlay-darker .stApp::before {{
            background: linear-gradient(180deg, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.6) 100%);
        }}

        .main, .block-container, .stBlock, .stMarkdown, 
        .stNumberInput, .stButton, .stMetric, .stTabs, 
        section[data-testid="stSidebar"], 
        div[data-testid="column"] {{
            position: relative !important;
            z-index: 2 !important;
            visibility: visible !important;
            opacity: 1 !important;
        }}

        input[type="number"], .stNumberInput, .stTextInput {{
            background: rgba(255, 255, 255, 0.15) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            visibility: visible !important;
            opacity: 1 !important;
        }}

        .stNumberInput > div > div > input {{
            background: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            visibility: visible !important;
        }}

        [data-testid="column"] {{
            background: rgba(0, 0, 0, 0.3) !important;
            padding: 20px !important;
            border-radius: 15px !important;
            backdrop-filter: blur(10px) !important;
        }}

        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            background: rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(8px) !important;
            padding: 8px !important;
            border-radius: 8px !important;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: #fff !important;
            text-shadow: 0 1px 10px rgba(0,0,0,0.45);
        }}

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

        @keyframes fadeInApp {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        .stSpinner > div {{
            border-top-color: #b388ff !important;
            border-right-color: #9b59ff !important;
            border-bottom-color: #8a2be2 !important;
            border-left-color: #7b1fa2 !important;
        }}
        </style>

        <video autoplay muted loop playsinline id="bg-video">
            <source src="{background_video_url}" type="video/mp4">
        </video>

        <script>
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

            if (document.querySelector('.stSpinner')) {{
                document.body.classList.add('overlay-darker');
            }}
        }})();
        </script>
    """, unsafe_allow_html=True)

# -------------------------------
# FIXED: Improved synthetic data generation with better distribution
# -------------------------------
def generate_data(n_samples=300):
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
    
    # FIXED: Better classification logic to ensure all three categories appear
    # Critical: High loan burden OR overspending
    critical_mask = (df["loan"] > df["income"] * 0.6) | (df["personal_exp"] + df["main_exp"] > df["income"] * 0.7)
    
    # Safe: Good investment ratio AND manageable expenses AND positive savings
    safe_mask = (df["investment"] > df["income"] * 0.15) & \
                (df["loan"] < df["income"] * 0.4) & \
                (df["savings"] > df["income"] * 0.1)
    
    # Moderate: Everything else
    df["status"] = "Moderate"
    df.loc[safe_mask, "status"] = "Safe"
    df.loc[critical_mask, "status"] = "Critical"
    
    # Generate stability scores based on status
    df.loc[df["status"] == "Critical", "stability_score"] = np.random.randint(20, 45, (df["status"] == "Critical").sum())
    df.loc[df["status"] == "Moderate", "stability_score"] = np.random.randint(45, 75, (df["status"] == "Moderate").sum())
    df.loc[df["status"] == "Safe", "stability_score"] = np.random.randint(75, 100, (df["status"] == "Safe").sum())
    
    return df

def train_models(df):
    X = df.drop(["status", "stability_score"], axis=1)
    y_class = df["status"]
    y_reg = df["stability_score"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(random_state=42, n_estimators=100).fit(X_scaled, y_class)
    reg = RandomForestRegressor(random_state=42, n_estimators=100).fit(X_scaled, y_reg)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    return scaler, clf, reg, kmeans

# -------------------------------
# FIXED: Improved user group classification
# -------------------------------
def determine_user_group(income, side_income, investment, personal_exp, main_exp, savings):
    """
    Classify user into one of three groups based on their actual financial behavior.
    """
    total_income = income + side_income
    if total_income == 0:
        return "Unknown"
    
    total_expenses = personal_exp + main_exp
    
    savings_rate = (savings / total_income * 100)
    investment_rate = (investment / total_income * 100)
    expense_rate = (total_expenses / total_income * 100)
    
    # Aggressive Investor: High investment rate (15%+) and decent savings
    if investment_rate >= 15 and savings_rate >= 10:
        return "Aggressive Investor"
    
    # High Spender: High expense rate or negative/very low savings
    elif expense_rate > 65 or savings_rate < 8:
        return "High Spender"
    
    # Balanced Saver: Moderate approach to all aspects
    else:
        return "Balanced Saver"

# -------------------------------
# FIXED: Enhanced recommendation logic with more detailed checks
# -------------------------------
def get_recommendations(values, financial_status):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    total_income = income + side_income
    recs = []
    
    # Loan recommendations
    loan_ratio = loan / income if income > 0 else 0
    if loan_ratio > 0.6:
        recs.append({"text": "🚨 Critical loan burden (>60% of income)! Prioritize debt reduction.", "type": "bad"})
    elif loan_ratio > 0.4:
        recs.append({"text": "⚠️ Moderate loan burden. Consider refinancing or increasing payments.", "type": "neutral"})
    else:
        recs.append({"text": "✅ Loan levels are healthy and manageable.", "type": "good"})
    
    # Investment recommendations
    inv_ratio = investment / total_income if total_income > 0 else 0
    if inv_ratio < 0.1:
        recs.append({"text": "📈 Investment too low (<10% of income). Increase for long-term wealth.", "type": "bad"})
    elif inv_ratio < 0.15:
        recs.append({"text": "📊 Investment is moderate. Consider increasing to 15-20% for better growth.", "type": "neutral"})
    else:
        recs.append({"text": "✅ Excellent investment rate! You're building wealth effectively.", "type": "good"})
    
    # Emergency fund recommendations
    emerg_ratio = emergency_exp / income if income > 0 else 0
    if emerg_ratio < 0.05:
        recs.append({"text": "🚨 Emergency fund critically low (<5% of income). Build it urgently.", "type": "bad"})
    elif emerg_ratio < 0.1:
        recs.append({"text": "⚠️ Emergency fund needs strengthening. Aim for 6 months of expenses.", "type": "neutral"})
    else:
        recs.append({"text": "✅ Emergency fund is solid and provides good security.", "type": "good"})
    
    # Expense recommendations
    expense_ratio = (personal_exp + main_exp) / total_income if total_income > 0 else 0
    if expense_ratio > 0.7:
        recs.append({"text": "💸 Expenses are too high (>70% of income). Cut unnecessary spending.", "type": "bad"})
    elif expense_ratio > 0.5:
        recs.append({"text": "💰 Expenses are moderate. Look for optimization opportunities.", "type": "neutral"})
    else:
        recs.append({"text": "✅ Expense management is excellent. Well done!", "type": "good"})
    
    # Savings recommendations
    savings_ratio = savings / total_income if total_income > 0 else 0
    if savings < 0:
        recs.append({"text": "🚨 Deficit budget! You're spending more than earning. Urgent action needed.", "type": "bad"})
    elif savings_ratio < 0.1:
        recs.append({"text": "⚠️ Low savings rate (<10%). Increase income or reduce expenses.", "type": "neutral"})
    else:
        recs.append({"text": "💰 Great savings rate! You're on track for financial security.", "type": "good"})
    
    # Overall status
    status_colors = {"Critical": "bad", "Moderate": "neutral", "Safe": "good"}
    recs.append({"text": f"🎯 Overall Financial Status: {financial_status}", "type": status_colors.get(financial_status, "neutral")})
    
    return recs

# -------------------------------
# FIXED: Enhanced financial assistant with better analysis
# -------------------------------
def financial_assistant(values, scaler, clf, reg, kmeans):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    
    # Get ML predictions
    scaled = scaler.transform([values])
    status = clf.predict(scaled)[0]
    score = reg.predict(scaled)[0]
    
    # Get behavior-based group
    group = determine_user_group(income, side_income, investment, personal_exp, main_exp, savings)
    
    return {
        "Financial Status": status,
        "Stability Score": round(max(0, min(100, score)), 2),  # Ensure score is between 0-100
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
        surplus = disposable_income - required_saving
        advice = f"✅ Goal is achievable! You'll have ₹{surplus:,.0f} left per month after saving."
    elif disposable_income > 0:
        shortfall = required_saving - disposable_income
        status = "Challenging"
        advice = f"⚠️ Tight budget. You need ₹{shortfall:,.0f} more per month. Consider reducing expenses or extending timeline."
    else:
        status = "Not Feasible"
        advice = f"🚨 Current expenses exceed income. Reduce spending by ₹{abs(disposable_income):,.0f}/month before setting savings goals."
    
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
    background_video_url = "https://cdn.pixabay.com/video/2022/08/16/128098-740186760_large.mp4"
    add_css(background_video_url)

    # Initialize models
    if 'models' not in st.session_state:
        df = generate_data()
        st.session_state.models = train_models(df)
        st.session_state.df = df  # Store for debugging if needed

    scaler, clf, reg, kmeans = st.session_state.models

    st.markdown("<h1 style='text-align:center; margin-bottom: 0.2rem;'>💰 Financial Health Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: rgba(255,255,255,0.85)'>Enter your <b>yearly</b> financial details (in ₹) below for comprehensive analysis.</p>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("📝 Your Financials")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Main Income (₹ per year)", min_value=0, value=400000, step=10000)
            side_income = st.number_input("Side Income (₹ per year)", min_value=0, value=20000, step=5000)
            annual_tax = st.number_input("Annual Tax (₹)", min_value=0, value=30000, step=5000)
            loan = st.number_input("Loan Payments (₹ per year)", min_value=0, value=80000, step=5000)
        with c2:
            investment = st.number_input("Investments (₹ per year)", min_value=0, value=70000, step=5000)
            personal_exp = st.number_input("Personal Expenses (₹ per year)", min_value=0, value=100000, step=5000)
            emergency_exp = st.number_input("Emergency Fund (₹)", min_value=0, value=25000, step=5000)
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
            status_emoji = {"Critical": "🚨", "Moderate": "⚠️", "Safe": "✅"}
            st.metric("Financial Status", f"{status_emoji.get(result['Financial Status'], '')} {result['Financial Status']}")
        with metric_cols[1]:
            st.metric("Stability Score", f"{result['Stability Score']}/100")
        with metric_cols[2]:
            st.metric("Financial Profile", result['Group'])
        with metric_cols[3]:
            savings_text = f"₹{savings:,.0f}"
            savings_delta = "positive" if savings > 0 else "negative"
            st.metric("Yearly Savings", savings_text, delta_color="normal" if savings > 0 else "inverse")

        tab1, tab2, tab3 = st.tabs(["📊 Expense Breakdown", "💡 Recommendations", "🎯 Goal Planner"])

        with tab1:
            st.subheader("Financial Allocation Overview")
            labels = ["Loan", "Investment", "Personal Exp.", "Emergency Fund", "Household Exp.", "Annual Tax"]
            sizes = [loan, investment, personal_exp, emergency_exp, main_exp, annual_tax]
            if savings > 0:
                labels.append("Savings")
                sizes.append(savings)
            df_pie = pd.DataFrame({"Category": labels, "Amount": sizes})
            fig = px.pie(df_pie, names='Category', values='Amount', hole=0.6,
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="white", size=14),
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0.3)")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary metrics
            total_income = income + side_income
            st.write("---")
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("Total Income", f"₹{total_income:,.0f}")
            with summary_cols[1]:
                total_expenses = annual_tax + loan + investment + personal_exp + emergency_exp + main_exp
                st.metric("Total Expenses", f"₹{total_expenses:,.0f}")
            with summary_cols[2]:
                savings_rate = (savings / total_income * 100) if total_income > 0 else 0
                st.metric("Savings Rate", f"{savings_rate:.1f}%")

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
                status_emoji_goal = {"Feasible": "✅", "Challenging": "⚠️", "Not Feasible": "🚨"}
                st.metric("Goal Status", f"{status_emoji_goal.get(plan['Status'], '')} {plan['Status']}")
            with plan_cols[1]:
                st.metric("Required Saving / Month", f"₹{plan['Required Saving per Month']:,.0f}")
            with plan_cols[2]:
                st.metric("Disposable Income / Month", f"₹{plan['Disposable Income per Month']:,.0f}")
            advice_type = "good" if plan['Status'] == 'Feasible' else ("neutral" if plan['Status'] == 'Challenging' else "bad")
            st.markdown(f"<div class='recommendation rec-{advice_type}'>💡 {plan['Advice']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()