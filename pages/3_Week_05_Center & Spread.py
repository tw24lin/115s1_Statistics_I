import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Week 5: Center & Spread", page_icon="🍎", layout="wide")

# ==========================================
# CSS INJECTION (Modern Streamlit Selectors)
# ==========================================
st.markdown("""
<style>
    /* Target the modern Streamlit tab button */
    button[data-testid="stTab"] {
        padding: 1.25rem 2rem !important;
    }
    
    /* Force all text elements inside the tab to enlarge and bold */
    button[data-testid="stTab"] * {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
    }
    
    /* Hover effect */
    button[data-testid="stTab"]:hover {
        background-color: #f0f2f6 !important;
        border-radius: 5px 5px 0px 0px !important;
    }
    
    /* Fallback for the parent container just in case */
    div[data-testid="stTabs"] button {
        padding: 1.25rem 2rem !important;
    }
    div[data-testid="stTabs"] button * {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING (Schema: N=1000)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('project_alpha.csv', parse_dates=['Date'])
        # Fallback rename just in case
        df = df.rename(columns={'Asset_Value': 'Closing_Price', 'Daily_Status': 'Daily_Trend', 'Intraday_Anomaly_Count': 'Volatility_Spikes'})
    except FileNotFoundError:
        # Use a local Random Generator to avoid breaking global seeds
        rng = np.random.default_rng(42)
        dates = pd.date_range(start='2020-01-01', periods=1000)
        # Create a slightly right-skewed distribution for realistic prices
        prices = rng.lognormal(mean=4.6, sigma=0.2, size=1000)
        trends = rng.choice(['Gain', 'Loss'], 1000, p=[0.55, 0.45])
        spikes = rng.poisson(2, 1000)
        df = pd.DataFrame({
            'Date': dates, 
            'Closing_Price': prices.round(2), 
            'Daily_Trend': trends, 
            'Volatility_Spikes': spikes
        })
    return df

df = load_data()

# ==========================================
# APP HEADER & DATA OVERVIEW
# ==========================================
st.title("Week 5 Intuition Engine: Center & Spread")

st.success("""
#### **Core Objective:** How do we summarize thousands of data points into a few meaningful numbers? 
###### In Week 5, we explore how to find the **Center** of our data, and how to measure its **Spread** (volatility). 
###### Use the interactive tools below to build your intuition.
""")

with st.expander("Step 1: Inspect the Raw Data (Overview)", expanded=False):
    st.markdown("We are continuing our analysis of **Project Alpha**. Here is our population data.")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(6), width='stretch')
    with col2:
        st.markdown(f"""
        **Total Records (N):** {len(df)} days
        
        **Variables:**
        * **Date:** When the measurement was taken.
        * **Closing_Price:** Quantitative (Continuous).
        * **Daily_Trend:** Categorical (Nominal).
        * **Volatility_Spikes:** Quantitative (Discrete).
        """)

main_tab1, main_tab2 = st.tabs(["Part 1: Measures of Center", "Part 2: Measures of Spread"])

# ==========================================
# PART 1: MEASURES OF CENTER
# ==========================================
with main_tab1:
    st.header("1. Measures of the Center of the Data")
    
    c_tab1, c_tab2, c_tab3, c_tab4 = st.tabs([
        "1.1 Mean, Median, Mode", 
        "1.2 Pop vs. Sample & Sigma", 
        "1.3 Geometric Mean", 
        "1.4 Symmetry & Skewness"
    ])
    
    # --- 1.1 Mean, Median, Mode ---
    with c_tab1:
        st.info("""💡##### **Concept:** The Mean (*average*), Median (*middle*), and Mode (*most frequent*) are different ways to define the 'center'.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            var_choice = st.selectbox("Select Variable to Analyze:", ['Closing_Price', 'Volatility_Spikes'])
            data_series = df[var_choice].dropna()
            calc_mean = data_series.mean()
            calc_median = data_series.median()
            
            if var_choice == 'Closing_Price':
                calc_mode = data_series.round(0).mode()[0]
                mode_note = "(Rounded to nearest dollar)"
            else:
                calc_mode = data_series.mode()[0]
                mode_note = ""
                
            st.markdown(f"""
            ### Center Metrics
            * **Mean:** {calc_mean:.2f}
            * **Median:** {calc_median:.2f}
            * **Mode:** {calc_mode:.2f} {mode_note}
            """)
            
            show_mean = st.checkbox("Show Mean Line", value=True)
            show_med = st.checkbox("Show Median Line", value=True)
            show_mode = st.checkbox("Show Mode Line", value=True)

        with col2:
            fig_center = px.histogram(df, x=var_choice, nbins=30, title=f"Distribution of {var_choice}")
            if show_mean:
                fig_center.add_vline(x=calc_mean, line_dash="dash", line_color="red", annotation_text="Mean")
            if show_med:
                fig_center.add_vline(x=calc_median, line_dash="dot", line_color="green", annotation_text="Median")
            if show_mode:
                fig_center.add_vline(x=calc_mode, line_dash="solid", line_color="orange", annotation_text="Mode")
            st.plotly_chart(fig_center, width='stretch')

    # --- 1.2 Sample vs Population (Notation) ---
    with c_tab2:
        st.info("💡##### **Concept:** We rarely have data for the entire **Population** ($N$). Instead, we take a **Sample** ($n$) and use the Sample Mean ($\\bar{x}$) to estimate the Population Mean ($\\mu$).")
        
        st.markdown("### 1. Demystifying the Formulas & Sigma ($\\Sigma$)")
        st.markdown("The formulas for both are identical in structure. The Greek letter **$\\Sigma$ (Sigma)** simply means **'add everything up'**.")
        
        c_sig1, c_sig2, c_sig3 = st.columns([1, 1, 1.2])
        with c_sig1:
            st.markdown("#### Population Mean")
            st.latex(r"\mu = \frac{\sum_{i=1}^{N} x_i}{N}")
        with c_sig2:
            st.markdown("#### Sample Mean")
            st.latex(r"\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}")
        with c_sig3:
            st.markdown("#### In Plain English:")
            st.markdown("1. Take every data point ($x_i$).\n2. Add them all together ($\\Sigma$).\n3. Divide by the total number of items ($N$ or $n$).")
            
        st.markdown("---")
        st.markdown("### 2. The Law of Large Numbers (Estimating $\\mu$)")
        
        pop_mean = df['Closing_Price'].mean()
        
        # Callback to update the seed BEFORE the page renders
        def generate_new_pop_sample():
            st.session_state['pop_sample_seed'] = np.random.randint(0, 10000)
            
        c_pop1, c_pop2 = st.columns([1, 2])
        with c_pop1:
            sample_size = st.slider("Select Sample Size (n):", min_value=5, max_value=1000, value=10, step=5)
            sample_df = df['Closing_Price'].sample(n=sample_size, random_state=st.session_state.get('pop_sample_seed', 42))
            sample_mean = sample_df.mean()
            
            st.markdown(f"""
            **True Pop. Mean ($N=1000$):** $\\mu = {pop_mean:.2f}$  
            **Your Sample Mean ($n={sample_size}$):** $\\bar{{x}} = {sample_mean:.2f}$  
            **Estimation Error:** `${abs(pop_mean - sample_mean):.2f}`
            """)
            
            # Using the on_click callback here!
            st.button("Draw a Different Random Sample", on_click=generate_new_pop_sample)

        with c_pop2:
            fig_samp = go.Figure()
            fig_samp.add_trace(go.Histogram(x=df['Closing_Price'], nbinsx=30, name='Full Population', marker_color='lightgray', opacity=0.5))
            fig_samp.add_trace(go.Histogram(x=sample_df, nbinsx=30, name=f'Sample (n={sample_size})', marker_color='blue', opacity=0.8))
            fig_samp.add_vline(x=pop_mean, line_dash="solid", line_color="black", line_width=3, annotation_text="μ (Pop Mean)")
            fig_samp.add_vline(x=sample_mean, line_dash="dash", line_color="red", line_width=3, annotation_text="x̄ (Sample Mean)")
            fig_samp.update_layout(barmode='overlay', title="Population vs. Sample Distribution")
            st.plotly_chart(fig_samp, width='stretch')

    # --- 1.3 Geometric Mean ---
    with c_tab3:
        st.info("💡##### **Concept:** Why is the Geometric Mean required in finance? Because the Arithmetic Mean **lies** about compounded growth.")
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown("*(Note: Returns are expressed in percentages (%). `100` means +100%, `-50` means -50%.)*")
            r1 = st.number_input("Year 1 Return (%):", value=100.0, step=10.0)
            r2 = st.number_input("Year 2 Return (%):", value=-50.0, step=10.0)
            
            arithmetic_mean = (r1 + r2) / 2
            m1 = 1 + (r1/100)
            m2 = 1 + (r2/100)
            if (m1 * m2) >= 0:
                geom_mean = (math.sqrt(m1 * m2) - 1) * 100
                geom_str = f"{geom_mean:.2f}%"
            else:
                geom_mean = -100
                geom_str = "Total Loss"
            
            st.markdown("#### The Math")
            st.markdown(f"**Arithmetic Average:** `({r1}% + {r2}%) / 2` = **{arithmetic_mean:.2f}%**")
            st.markdown(f"**Geometric Average:** `sqrt({m1} * {m2}) - 1` = **{geom_str}**")
            
        with c2:
            st.markdown("#### The Reality Check ($100 Investment)")
            actual_y0 = 100
            actual_y1 = actual_y0 * m1
            actual_y2 = actual_y1 * m2
            
            arith_multiplier = 1 + (arithmetic_mean/100)
            fake_y0 = 100
            fake_y1 = fake_y0 * arith_multiplier
            fake_y2 = fake_y1 * arith_multiplier
            
            sim_df = pd.DataFrame({
                'Year': ['Start', 'Yr 1', 'Yr 2'],
                'Actual Portfolio (Geometric)': [actual_y0, actual_y1, actual_y2],
                'The Arithmetic Fantasy': [fake_y0, fake_y1, fake_y2]
            })
            
            fig_geom = go.Figure()
            fig_geom.add_trace(go.Scatter(x=sim_df['Year'], y=sim_df['Actual Portfolio (Geometric)'], mode='lines+markers', name='Actual Dollar Value', line=dict(color='green', width=4)))
            fig_geom.add_trace(go.Scatter(x=sim_df['Year'], y=sim_df['The Arithmetic Fantasy'], mode='lines+markers', name='Arithmetic Implied', line=dict(color='red', width=4, dash='dash')))
            fig_geom.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="Breakeven ($100)")
            st.plotly_chart(fig_geom, width='stretch')
            
            if arithmetic_mean > geom_mean:
                st.warning(f"##### **Look at the red line! The Arithmetic mean claims you are making +{arithmetic_mean:.2f}% every year. But the green line shows your true portfolio value. The Geometric mean ({geom_str}) tells the exact truth about your actual compounded growth.**")

    # --- 1.4 Symmetry & Skewness ---
    with c_tab4:
        st.info("💡##### **Concept:** Skewness occurs when outliers pull the Mean away from the Median.")
        skew_type = st.radio("Select Shape:", ["Symmetric (Normal)", "Right-Skewed (Positive Skew)", "Left-Skewed (Negative Skew)"], horizontal=True)
        
        # Use isolated random generator to prevent global seed resets
        rng_skew = np.random.default_rng(1)
        if skew_type == "Symmetric (Normal)":
            skew_data = rng_skew.normal(100, 10, 1000)
        elif skew_type == "Right-Skewed (Positive Skew)":
            skew_data = rng_skew.lognormal(mean=4.5, sigma=0.5, size=1000)
        else:
            skew_data = 200 - rng_skew.lognormal(mean=4.5, sigma=0.5, size=1000)
            
        skew_df = pd.DataFrame({'Value': skew_data})
        s_mean, s_med, s_mode = skew_df['Value'].mean(), skew_df['Value'].median(), skew_df['Value'].round(0).mode()[0]
        
        c1, c2 = st.columns([1, 2.5])
        with c1:
            st.markdown("### The 'Pull'")
            if skew_type == "Symmetric (Normal)": 
                st.success("**Mean $\\approx$ Median $\\approx$ Mode**")
            elif skew_type == "Right-Skewed (Positive Skew)": 
                st.warning("##### **Mean > Median > Mode (Pulled Right)**")
            else: 
                st.error("**Mean < Median < Mode** (Pulled Left)")
            st.markdown(f"- **Mean:** {s_mean:.1f}\n- **Median:** {s_med:.1f}\n- **Mode:** {s_mode:.1f}")
            
        with c2:
            fig_skew = px.histogram(skew_df, x='Value', nbins=50)
            fig_skew.add_vline(x=s_mean, line_dash="dash", line_color="red", annotation_text="Mean")
            fig_skew.add_vline(x=s_med, line_dash="dot", line_color="green", annotation_text="Median")
            st.plotly_chart(fig_skew, width='stretch')

# ==========================================
# PART 2: MEASURES OF SPREAD
# ==========================================
with main_tab2:
    st.header("2. Measures of the Spread of the Data")
    
    s_tab1, s_tab2, s_tab3, s_tab4, s_tab5 = st.tabs([
        "2.1 The 'n-1' Intuition", 
        "2.2 Types of Variability", 
        "2.3 Apples vs. Oranges (Z-Scores)", 
        "2.4 Coefficient of Variation",
        "2.5 Chebyshev vs. Empirical"
    ])
    
    # --- 2.1 The n-1 Intuition ---
    with s_tab1:
        st.info("💡##### **Concept:** Why do we divide by $n-1$ for Sample Variance? Because a sample usually clumps together, causing us to *underestimate* the true Population spread. Dividing by a smaller number ($n-1$) artificially inflates the variance to fix this bias.")
        
        c1, c2 = st.columns([1.5, 2])
        
        # True population setup (Using isolated generator)
        rng_pop = np.random.default_rng(42)
        pop_array = rng_pop.normal(100, 15, 20)
        pop_mean = np.mean(pop_array)
        true_pop_variance = np.var(pop_array, ddof=0)
        
        # Callback to update the sample BEFORE the page renders
        def draw_n3_sample():
            # Use the global numpy space here so it truly randomizes on every single click
            st.session_state['n1_sample'] = np.random.choice(pop_array, 3, replace=False)
            
        if 'n1_sample' not in st.session_state:
            st.session_state['n1_sample'] = np.random.choice(pop_array, 3, replace=False)
            
        with c1:
            st.markdown(f"**True Population Mean ($\\mu$):** `{pop_mean:.2f}`")
            st.markdown(f"**True Population Variance ($\\sigma^2$):** `{true_pop_variance:.2f}`")
            st.markdown("---")
            st.markdown("Draw a small sample of $n=3$ to see what happens to our spread calculations.")
            
            # Using the on_click callback here!
            st.button("Draw Sample (n=3)", on_click=draw_n3_sample)
                
            samp = st.session_state['n1_sample']
            samp_mean = np.mean(samp)
            
            # The calculation breakdown
            sum_sq_diff = sum((x - samp_mean)**2 for x in samp)
            var_divide_n = sum_sq_diff / 3       # Biased
            var_divide_n_minus_1 = sum_sq_diff / 2 # Corrected
            
            st.markdown(f"**Your Sample Mean ($\\bar{{x}}$):** `{samp_mean:.2f}`")
            
            st.markdown("#### Calculating the Variance")
            st.markdown(f"Dividing by $n$ (Biased): `{var_divide_n:.2f}`")
            st.markdown(f"Dividing by $n-1$ (Corrected): `{var_divide_n_minus_1:.2f}`")
            
            if var_divide_n < true_pop_variance:
                st.error(f"Notice how dividing by $n$ underestimated the true population variance ({var_divide_n:.2f} < {true_pop_variance:.2f})! The $n-1$ correction pushed it closer to the truth.")
            else:
                st.warning(f"##### **In this specific random draw, your sample points happened to be very spread out, so the $n$ variance was larger than the population variance ({var_divide_n:.2f} > {true_pop_variance:.2f}). However, *on average* across many samples, dividing by $n$ underestimates the true spread, which is why we must use $n-1$!**")
                
        with c2:
            st.markdown("### The 'Clumping' Effect")
            st.markdown("Notice how the blue sample points are usually clustered closer to their *own* sample mean (red dashed line) than they are to the true population mean (black line).")
            
            fig_n1 = go.Figure()
            # Plot population
            fig_n1.add_trace(go.Scatter(x=pop_array, y=[1]*len(pop_array), mode='markers', marker=dict(color='lightgray', size=10), name='Population'))
            fig_n1.add_vline(x=pop_mean, line_width=2, line_color="black", annotation_text="Pop Mean")
            
            # Plot Sample
            fig_n1.add_trace(go.Scatter(x=samp, y=[1]*len(samp), mode='markers', marker=dict(color='blue', size=16), name='Sample (n=3)'))
            fig_n1.add_vline(x=samp_mean, line_dash="dash", line_color="red", line_width=2, annotation_text="Sample Mean")
            
            fig_n1.update_layout(yaxis=dict(showticklabels=False, range=[0.5, 1.5]), xaxis_title="Value", height=300)
            st.plotly_chart(fig_n1, width='stretch')

    # --- 2.2 Types of Variability ---
    with s_tab2:
        st.info("💡##### **Concept:** Not all spread is created equal. We must understand *why* data varies to analyze it properly.")
        
        var_type = st.selectbox("Select a Scenario to Simulate:", [
            "Measurement Variability (Sensor Noise)", 
            "Natural Variability (Inherent Differences)", 
            "Induced Variability (A/B Testing)", 
            "Sample Variability (Random Batches)"
        ])
        
        st.markdown("---")
        
        c1, c2 = st.columns([1, 2])
        if "Measurement" in var_type:
            with c1:
                st.markdown("### Measurement Variability")
                st.markdown("We are measuring the **exact same object** (True Weight = 100g) over and over. The variability comes from the inaccuracy of the scale.")
                noise = st.slider("Scale Inaccuracy (Noise)", 0.5, 10.0, 2.0)
            with c2:
                rng_meas = np.random.default_rng(42)
                meas_data = rng_meas.normal(100, noise, 200)
                fig_v = px.histogram(x=meas_data, nbins=30, title=f"200 Scans of the Same 100g Object (Spread = {noise})")
                fig_v.update_layout(xaxis_title="Measured Weight (g)")
                st.plotly_chart(fig_v, width='stretch')
                
        elif "Natural" in var_type:
            with c1:
                st.markdown("### Natural Variability")
                st.markdown("We are measuring **different objects** from the same population. Nature naturally produces different sizes.")
                st.markdown("No slider needed here: this spread exists in nature without any intervention or measurement error.")
            with c2:
                rng_nat = np.random.default_rng(43)
                nat_data = rng_nat.normal(150, 15, 500)
                fig_v = px.histogram(x=nat_data, nbins=30, title="Heights of 500 Different Oak Trees")
                fig_v.update_layout(xaxis_title="Tree Height (cm)")
                st.plotly_chart(fig_v, width='stretch')
                
        elif "Induced" in var_type:
            with c1:
                st.markdown("### Induced Variability")
                st.markdown("We intentionally introduce a change (a treatment) to see if it induces a difference in the spread.")
                lift = st.slider("Marketing Campaign Effect (+ sales)", 0, 50, 20)
            with c2:
                rng_ind = np.random.default_rng(44)
                control = rng_ind.normal(100, 10, 300)
                treatment = rng_ind.normal(100 + lift, 10, 300)
                fig_v = go.Figure()
                fig_v.add_trace(go.Histogram(x=control, name="Control Group (No Ad)", opacity=0.7))
                fig_v.add_trace(go.Histogram(x=treatment, name="Treatment Group (Saw Ad)", opacity=0.7))
                fig_v.update_layout(barmode='overlay', title="A/B Testing Distributions", xaxis_title="Sales ($)")
                st.plotly_chart(fig_v, width='stretch')
                
        elif "Sample" in var_type:
            # Callback to update the seed for 5 random samples BEFORE page renders
            def pull_5_samples():
                st.session_state['samp_var_seed'] = np.random.randint(0, 10000)
                
            with c1:
                st.markdown("### Sample Variability")
                st.markdown("If we repeatedly pull samples from the same population, the **Sample Mean** itself will vary slightly every time by pure chance.")
                
                # Using the on_click callback here!
                st.button("Pull 5 New Random Samples", on_click=pull_5_samples)
                
            with c2:
                rng_samp = np.random.default_rng(st.session_state.get('samp_var_seed', 1))
                pop = rng_samp.normal(50, 10, 10000)
                means = [np.mean(rng_samp.choice(pop, 30)) for _ in range(5)]
                
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=means, y=[1, 2, 3, 4, 5], mode='markers', marker=dict(size=15, color='orange')))
                fig_v.add_vline(x=50, line_dash='dash', line_color='black', annotation_text='True Pop Mean (50)')
                fig_v.update_layout(yaxis=dict(showticklabels=False), title="5 Different Sample Means (They aren't exactly 50!)", xaxis_title="Calculated Sample Mean")
                st.plotly_chart(fig_v, width='stretch')

    # --- 2.3 Comparing Apples to Oranges (Z-Scores) ---
    with s_tab3:
        st.info("💡##### **Concept:** A z-score standardizes data. It allows us to literally compare 'Apples to Oranges' by measuring how far a value is from its *own* mean, using its *own* standard deviation as the ruler.")
        
        st.markdown("### Which fruit is more exceptionally heavy?")
        
        c1, c2, c3 = st.columns([1, 1, 1.5])
        
        # Define populations
        apple_mu, apple_sig = 150, 15
        orange_mu, orange_sig = 200, 25
        
        with c1:
            st.markdown("🍎 **Apples**")
            st.markdown(f"Avg ($\\mu$) = {apple_mu}g | StdDev ($\\sigma$) = {apple_sig}g")
            apple_x = st.slider("Your Apple's Weight ($x$)", 100, 220, 185)
            z_apple = (apple_x - apple_mu) / apple_sig
            st.markdown("**Apple Z-Score:**")
            st.latex(f"z = \\frac{{{apple_x} - {apple_mu}}}{{{apple_sig}}} = {z_apple:.2f}")
            
        with c2:
            st.markdown("🍊 **Oranges**")
            st.markdown(f"Avg ($\\mu$) = {orange_mu}g | StdDev ($\\sigma$) = {orange_sig}g")
            orange_x = st.slider("Your Orange's Weight ($x$)", 100, 300, 235)
            z_orange = (orange_x - orange_mu) / orange_sig
            st.markdown("**Orange Z-Score:**")
            st.latex(f"z = \\frac{{{orange_x} - {orange_mu}}}{{{orange_sig}}} = {z_orange:.2f}")
            
        with c3:
            st.markdown("### The Standardized Comparison")
            if z_apple > z_orange:
                st.success(f"The **Apple** is relatively heavier! Even though {apple_x}g is less than {orange_x}g, the apple is further above average for its species.")
            elif z_orange > z_apple:
                st.success(f"The **Orange** is relatively heavier! It is further above the orange average than the apple is above the apple average.")
            else:
                st.success("They are equally exceptionally heavy relative to their species!")
                
            # Plot on Standard Normal
            x_norm = np.linspace(-4, 4, 200)
            y_norm = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_norm ** 2))
            
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', line=dict(color='lightgray'), name='N(0,1)'))
            fig_z.add_vline(x=0, line_dash='dash', line_color='black', annotation_text='Mean (0)')
            
            fig_z.add_trace(go.Scatter(x=[z_apple], y=[0.05], mode='markers', marker=dict(color='red', size=15), name='Your Apple'))
            fig_z.add_trace(go.Scatter(x=[z_orange], y=[0.1], mode='markers', marker=dict(color='orange', size=15), name='Your Orange'))
            
            fig_z.update_layout(xaxis_title="Z-Score (Standard Deviations away from Mean)", yaxis_title="")
            st.plotly_chart(fig_z, width='stretch')

    # --- 2.4 Coefficient of Variation ---
    with s_tab4:
        st.info("💡##### **Concept:** The Coefficient of Variation ($CV = \\frac{\\sigma}{\\mu}$) measures *relative* variability. A \\$10 standard deviation means nothing until you know if the asset costs \\$50 or \\$5,000.")
        
        st.markdown("### Comparing Two Assets")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Asset A: Penny Stock**")
            mean_a = st.number_input("Asset A Mean Price:", value=5.0)
            std_a = st.number_input("Asset A Std Dev:", value=2.0)
            cv_a = (std_a / mean_a) * 100
            st.markdown(f"**CV:** `( {std_a} / {mean_a} ) * 100` = **{cv_a:.1f}%**")
            
        with c2:
            st.markdown("**Asset B: Blue Chip Stock**")
            mean_b = st.number_input("Asset B Mean Price:", value=500.0)
            std_b = st.number_input("Asset B Std Dev:", value=50.0)
            cv_b = (std_b / mean_b) * 100
            st.markdown(f"**CV:** `( {std_b} / {mean_b} ) * 100` = **{cv_b:.1f}%**")
            
        st.markdown("---")
        if cv_a > cv_b:
            st.warning("##### **Even though Asset B has a higher *absolute* Standard Deviation, Asset A has higher *relative* volatility (CV). Asset A is riskier relative to its price.**")
        else:
            st.warning("##### **Asset B is riskier relative to its price.**")

    # --- 2.5 Chebyshev vs Empirical ---
    with s_tab5:
        st.info("💡##### **Concept:** How much data falls within $k$ standard deviations of the mean? The **Empirical Rule** applies only to bell-shaped data. **Chebyshev's Rule** applies to *any* distribution shape.")
        
        # Calculate actual empirical data from our dataset
        mu = df['Closing_Price'].mean()
        sig = df['Closing_Price'].std(ddof=0)
        
        within_1 = len(df[(df['Closing_Price'] >= mu - sig) & (df['Closing_Price'] <= mu + sig)]) / len(df) * 100
        within_2 = len(df[(df['Closing_Price'] >= mu - 2*sig) & (df['Closing_Price'] <= mu + 2*sig)]) / len(df) * 100
        within_3 = len(df[(df['Closing_Price'] >= mu - 3*sig) & (df['Closing_Price'] <= mu + 3*sig)]) / len(df) * 100
        
        col1, col2 = st.columns([1.5, 2])
        with col1:
            st.markdown("### The Rules")
            k_val = st.radio("Select $k$ (Standard Deviations):", [1, 2, 3], index=1)
            
            if k_val == 1:
                emp_target = "Approx 68%"
                cheb_target = "At least 0% (Rule doesn't apply for k=1)"
            elif k_val == 2:
                emp_target = "Approx 95%"
                cheb_target = "At least 75% ($1 - 1/2^2$)"
            else:
                emp_target = "Approx 99.7%"
                cheb_target = "At least 88.9% ($1 - 1/3^2$)"
                
            actual_val = within_1 if k_val == 1 else (within_2 if k_val == 2 else within_3)
            
            st.markdown("#### Chebyshev's Formula")
            st.latex(r"\text{Minimum \%} = \left(1 - \frac{1}{k^2}\right) \times 100")

            st.markdown(f"""
            * **Empirical Rule Expectation:** {emp_target}
            * **Chebyshev's Expectation:** {cheb_target}
            * **Actual in our Dataset:** **{actual_val:.1f}%**
            """)
            
            st.markdown("*(Our dataset is slightly right-skewed, so it may not perfectly match the Empirical Rule!)*")
            
        with col2:
            fig_rules = px.histogram(df, x='Closing_Price', nbins=40, title="Visualizing the Spread boundaries")
            fig_rules.add_vline(x=mu, line_color='black', annotation_text='Mean')
            
            # Draw boundaries
            fig_rules.add_vrect(x0=mu - (k_val * sig), x1=mu + (k_val * sig), 
                                fillcolor="red", opacity=0.2, line_width=0, 
                                annotation_text=f"± {k_val} Std Devs")
            
            st.plotly_chart(fig_rules, width='stretch')