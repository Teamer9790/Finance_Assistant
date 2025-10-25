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
# IMPROVED: Generate balanced training data with clear patterns
# -------------------------------
def generate_data(n_samples=1000):
    """
    Generate synthetic training data with CLEAR patterns for ML to learn
    """
    np.random.seed(42)
    
    # Create empty lists to store data for each class
    all_data = []
    
    # Generate SAFE profiles (33% of data)
    n_safe = n_samples // 3
    for _ in range(n_safe):
        income = np.random.randint(400000, 1000000)
        side_income = np.random.randint(20000, 150000)
        total_income = income + side_income
        
        # Safe characteristics: Good investment, low loan, good savings
        investment = int(total_income * np.random.uniform(0.15, 0.25))  # 15-25% investment
        loan = int(income * np.random.uniform(0.10, 0.35))  # 10-35% loan
        annual_tax = int(income * np.random.uniform(0.05, 0.12))
        emergency_exp = int(income * np.random.uniform(0.08, 0.15))
        
        # Keep expenses low for good savings
        personal_exp = int(total_income * np.random.uniform(0.15, 0.25))
        main_exp = int(total_income * np.random.uniform(0.15, 0.25))
        
        savings = total_income - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)
        stability_score = np.random.randint(75, 95)
        
        all_data.append({
            'income': income,
            'side_income': side_income,
            'annual_tax': annual_tax,
            'loan': loan,
            'investment': investment,
            'personal_exp': personal_exp,
            'emergency_exp': emergency_exp,
            'main_exp': main_exp,
            'savings': savings,
            'status': 'Safe',
            'stability_score': stability_score
        })
    
    # Generate MODERATE profiles (33% of data)
    n_moderate = n_samples // 3
    for _ in range(n_moderate):
        income = np.random.randint(300000, 800000)
        side_income = np.random.randint(10000, 100000)
        total_income = income + side_income
        
        # Moderate characteristics: Average everything
        investment = int(total_income * np.random.uniform(0.08, 0.15))  # 8-15% investment
        loan = int(income * np.random.uniform(0.35, 0.55))  # 35-55% loan
        annual_tax = int(income * np.random.uniform(0.05, 0.12))
        emergency_exp = int(income * np.random.uniform(0.03, 0.08))
        
        # Moderate expenses
        personal_exp = int(total_income * np.random.uniform(0.20, 0.35))
        main_exp = int(total_income * np.random.uniform(0.20, 0.35))
        
        savings = total_income - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)
        stability_score = np.random.randint(45, 75)
        
        all_data.append({
            'income': income,
            'side_income': side_income,
            'annual_tax': annual_tax,
            'loan': loan,
            'investment': investment,
            'personal_exp': personal_exp,
            'emergency_exp': emergency_exp,
            'main_exp': main_exp,
            'savings': savings,
            'status': 'Moderate',
            'stability_score': stability_score
        })
    
    # Generate CRITICAL profiles (34% of data)
    n_critical = n_samples - n_safe - n_moderate
    for _ in range(n_critical):
        income = np.random.randint(200000, 600000)
        side_income = np.random.randint(0, 50000)
        total_income = income + side_income
        
        # Critical characteristics: High loan OR high expenses OR low investment
        investment = int(total_income * np.random.uniform(0.02, 0.08))  # 2-8% investment (low)
        loan = int(income * np.random.uniform(0.55, 0.80))  # 55-80% loan (high)
        annual_tax = int(income * np.random.uniform(0.05, 0.12))
        emergency_exp = int(income * np.random.uniform(0.01, 0.05))  # Low emergency fund
        
        # High expenses
        personal_exp = int(total_income * np.random.uniform(0.30, 0.50))
        main_exp = int(total_income * np.random.uniform(0.25, 0.45))
        
        savings = total_income - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)
        stability_score = np.random.randint(20, 45)
        
        all_data.append({
            'income': income,
            'side_income': side_income,
            'annual_tax': annual_tax,
            'loan': loan,
            'investment': investment,
            'personal_exp': personal_exp,
            'emergency_exp': emergency_exp,
            'main_exp': main_exp,
            'savings': savings,
            'status': 'Critical',
            'stability_score': stability_score
        })
    
    df = pd.DataFrame(all_data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def train_models(df):
    """
    Train ML models with proper hyperparameters
    """
    X = df.drop(["status", "stability_score"], axis=1)
    y_class = df["status"]
    y_reg = df["stability_score"]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest Classifier with balanced weights
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle any class imbalance
    ).fit(X_scaled, y_class)
    
    # Train Random Forest Regressor
    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ).fit(X_scaled, y_reg)
    
    # Train K-Means for clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    
    return scaler, clf, reg, kmeans, df

# -------------------------------
# User group classification using K-Means
# -------------------------------
def determine_user_group(cluster_id, savings_ratio, investment_ratio, expense_ratio):
    """
    Map K-Means cluster to meaningful user groups
    """
    # Analyze the dominant characteristic
    if investment_ratio >= 15:
        return "Aggressive Investor"
    elif expense_ratio > 60 or savings_ratio < 10:
        return "High Spender"
    else:
        return "Balanced Saver"

# -------------------------------
# Enhanced recommendation logic
# -------------------------------
def get_recommendations(values, financial_status):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    total_income = income + side_income
    recs = []
    
    # Loan recommendations
    loan_ratio = loan / income if income > 0 else 0
    if loan_ratio > 0.60:
        recs.append({"text": "🚨 Critical loan burden (>60% of income)! Prioritize debt reduction immediately.", "type": "bad"})
    elif loan_ratio > 0.35:
        recs.append({"text": "⚠️ Moderate loan burden. Consider refinancing or increasing payments.", "type": "neutral"})
    else:
        recs.append({"text": "✅ Loan levels are healthy and manageable.", "type": "good"})
    
    # Investment recommendations
    inv_ratio = investment / total_income if total_income > 0 else 0
    if inv_ratio < 0.10:
        recs.append({"text": "📈 Investment too low (<10% of income). Increase for long-term wealth building.", "type": "bad"})
    elif inv_ratio < 0.15:
        recs.append({"text": "📊 Investment is moderate. Consider increasing to 15-20% for better growth.", "type": "neutral"})
    else:
        recs.append({"text": "✅ Excellent investment rate (≥15%)! You're building wealth effectively.", "type": "good"})
    
    # Emergency fund recommendations
    emerg_ratio = emergency_exp / income if income > 0 else 0
    if emerg_ratio < 0.05:
        recs.append({"text": "🚨 Emergency fund critically low (<5% of income). Build it urgently.", "type": "bad"})
    elif emerg_ratio < 0.10:
        recs.append({"text": "⚠️ Emergency fund needs strengthening. Aim for 3-6 months of expenses.", "type": "neutral"})
    else:
        recs.append({"text": "✅ Emergency fund is solid and provides good security.", "type": "good"})
    
    # Expense recommendations
    expense_ratio = (personal_exp + main_exp) / total_income if total_income > 0 else 0
    if expense_ratio > 0.70:
        recs.append({"text": "💸 Expenses are too high (>70% of income). Cut unnecessary spending urgently.", "type": "bad"})
    elif expense_ratio > 0.50:
        recs.append({"text": "💰 Expenses are moderate-high. Look for optimization opportunities.", "type": "neutral"})
    else:
        recs.append({"text": "✅ Expense management is excellent (≤50%). Well done!", "type": "good"})
    
    # Savings recommendations
    savings_ratio = savings / total_income if total_income > 0 else 0
    if savings < 0:
        recs.append({"text": "🚨 Deficit budget! You're spending more than earning. Urgent action needed.", "type": "bad"})
    elif savings_ratio < 0.10:
        recs.append({"text": "⚠️ Low savings rate (<10%). Increase income or reduce expenses.", "type": "neutral"})
    elif savings_ratio < 0.15:
        recs.append({"text": "💰 Good savings rate (10-15%). Keep it up!", "type": "neutral"})
    else:
        recs.append({"text": "✅ Excellent savings rate (≥15%)! You're on track for financial security.", "type": "good"})
    
    # Overall status with detailed message
    status_messages = {
        "Safe": "🎯 ML Model Prediction: SAFE ✅ - Your finances are in excellent shape!",
        "Moderate": "🎯 ML Model Prediction: MODERATE ⚠️ - You're doing okay, but there's room for improvement.",
        "Critical": "🎯 ML Model Prediction: CRITICAL 🚨 - Immediate action needed to improve your financial health."
    }
    status_colors = {"Critical": "bad", "Moderate": "neutral", "Safe": "good"}
    recs.append({"text": status_messages.get(financial_status, f"🎯 Overall Status: {financial_status}"), 
                 "type": status_colors.get(financial_status, "neutral")})
    
    return recs

# -------------------------------
# Financial assistant with ML predictions
# -------------------------------
def financial_assistant(values, scaler, clf, reg, kmeans):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    
    # Scale the input
    scaled = scaler.transform([values])
    
    # Get ML predictions
    status = clf.predict(scaled)[0]
    status_proba = clf.predict_proba(scaled)[0]
    score = reg.predict(scaled)[0]
    cluster = kmeans.predict(scaled)[0]
    
    # Calculate ratios for user group
    total_income = income + side_income
    savings_ratio = (savings / total_income * 100) if total_income > 0 else 0
    investment_ratio = (investment / total_income * 100) if total_income > 0 else 0
    expense_ratio = ((personal_exp + main_exp) / total_income * 100) if total_income > 0 else 0
    
    group = determine_user_group(cluster, savings_ratio, investment_ratio, expense_ratio)
    
    return {
        "Financial Status": status,
        "Stability Score": round(max(0, min(100, score)), 2),
        "Group": group,
        "Recommendations": get_recommendations(values, status),
        "Confidence": {
            "Critical": round(status_proba[0] * 100, 1),
            "Moderate": round(status_proba[1] * 100, 1),
            "Safe": round(status_proba[2] * 100, 1)
        }
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

    # Train models once and cache
    if 'models' not in st.session_state:
        with st.spinner('🤖 Training ML models on 1000 financial profiles...'):
            df = generate_data(n_samples=1000)
            scaler, clf, reg, kmeans, training_df = train_models(df)
            st.session_state.models = (scaler, clf, reg, kmeans)
            st.session_state.training_df = training_df
            st.success('✅ ML Models trained successfully!')

    scaler, clf, reg, kmeans = st.session_state.models

    st.markdown("<h1 style='text-align:center; margin-bottom: 0.2rem;'>💰 Financial Health Assistant (ML-Powered)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: rgba(255,255,255,0.85)'>Using Random Forest ML models trained on 1000 financial profiles</p>", unsafe_allow_html=True)
    
    # Show model stats
    with st.expander("🤖 View ML Model Statistics"):
        df = st.session_state.training_df
        st.write("**Training Data Distribution:**")
        status_counts = df['status'].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Safe Profiles", status_counts.get('Safe', 0))
        with col2:
            st.metric("Moderate Profiles", status_counts.get('Moderate', 0))
        with col3:
            st.metric("Critical Profiles", status_counts.get('Critical', 0))
        
        st.write("**Model Details:**")
        st.write(f"- Classifier: Random Forest with 200 trees")
        st.write(f"- Features: 9 financial parameters")
        st.write(f"- Classes: Safe, Moderate, Critical")
    
    st.write("")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("📝 Your Financials")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Main Income (₹/year)", min_value=0, value=500000, step=10000)
            side_income = st.number_input("Side Income (₹/year)", min_value=0, value=50000, step=5000)
            annual_tax = st.number_input("Annual Tax (₹)", min_value=0, value=30000, step=5000)
            loan = st.number_input("Loan Payments (₹/year)", min_value=0, value=150000, step=5000)
        with c2:
            investment = st.number_input("Investments (₹/year)", min_value=0, value=100000, step=5000)
            personal_exp = st.number_input("Personal Expenses (₹/year)", min_value=0, value=100000, step=5000)
            emergency_exp = st.number_input("Emergency Fund (₹)", min_value=0, value=35000, step=5000)
            main_exp = st.number_input("Household Expenses (₹/year)", min_value=0, value=100000, step=5000)

    with col2:
        st.header("🎯 Your Goals")
        goal_amount = st.number_input("Goal amount (₹)", min_value=0, value=50000, step=5000)
        time_period = st.number_input("Time period (months)", min_value=1, value=6, step=1)
        st.write("")
        analyze_button = st.button("🔍 Analyze with ML Model", type="primary")

    st.divider()

    if analyze_button:
        with st.spinner('🤖 Running ML predictions...'):
            savings = (income + side_income) - (annual_tax + loan + investment + personal_exp + emergency_exp + main_exp)
            values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
            result = financial_assistant(values, scaler, clf, reg, kmeans)
            plan = goal_saving_plan(goal_amount, time_period, income, side_income, annual_tax, loan, personal_exp, emergency_exp, main_exp)

        st.header("📈 ML-Powered Financial Analysis")
        
        # Show prediction confidence
        conf_cols = st.columns([1, 2, 1])
        with conf_cols[1]:
            st.write("**ML Model Confidence:**")
            conf_df = pd.DataFrame({
                'Status': ['Critical', 'Moderate', 'Safe'],
                'Confidence': [result['Confidence']['Critical'], 
                             result['Confidence']['Moderate'], 
                             result['Confidence']['Safe']]
            })
            fig_conf = px.bar(conf_df, x='Status', y='Confidence', 
                            color='Status',
                            color_discrete_map={'Critical': '#e74c3c', 'Moderate': '#f39c12', 'Safe': '#2ecc71'})
            fig_conf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0.2)", 
                font=dict(color="white"),
                showlegend=False,
                height=250
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        st.write("")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            status_emoji = {"Critical": "🚨", "Moderate": "⚠️", "Safe": "✅"}
            st.metric("ML Prediction", f"{status_emoji.get(result['Financial Status'], '')} {result['Financial Status']}")
        with metric_cols[1]:
            st.metric("Stability Score", f"{result['Stability Score']}/100")
        with metric_cols[2]:
            st.metric("User Profile", result['Group'])
        with metric_cols[3]:
            st.metric("Yearly Savings", f"₹{savings:,.0f}")

        tab1, tab2, tab3 = st.tabs(["📊 Expense Breakdown", "💡 ML Recommendations", "🎯 Goal Planner"])

        with tab1:
            st.subheader("Financial Allocation Overview")
            labels = ["Loan", "Investment", "Personal", "Emergency", "Household", "Tax"]
            sizes = [loan, investment, personal_exp, emergency_exp, main_exp, annual_tax]
            if savings > 0:
                labels.append("Savings")
                sizes.append(savings)
            df_pie = pd.DataFrame({"Category": labels, "Amount": sizes})
            fig = px.pie(df_pie, names='Category', values='Amount', hole=0.6,
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="white", size=14),
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0.3)")
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("AI-Powered Recommendations")
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
                        st_lottie(anim, height=54, key=f"rec_{i}")
                with cols[1]:
                    st.markdown(f"<div class='recommendation rec-{rec['type']}'>{rec['text']}</div>", unsafe_allow_html=True)

        with tab3:
            st.subheader(f"Goal: Save ₹{goal_amount:,.0f} in {time_period} months")
            plan_cols = st.columns(3)
            with plan_cols[0]:
                status_emoji_goal = {"Feasible": "✅", "Challenging": "⚠️", "Not Feasible": "🚨"}
                st.metric("Status", f"{status_emoji_goal.get(plan['Status'], '')} {plan['Status']}")
            with plan_cols[1]:
                st.metric("Required/Month", f"₹{plan['Required Saving per Month']:,.0f}")
            with plan_cols[2]:
                st.metric("Available/Month", f"₹{plan['Disposable Income per Month']:,.0f}")
            advice_type = "good" if plan['Status'] == 'Feasible' else ("neutral" if plan['Status'] == 'Challenging' else "bad")
            st.markdown(f"<div class='recommendation rec-{advice_type}'>💡 {plan['Advice']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()