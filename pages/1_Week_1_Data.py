import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Chapter 1: Sampling and Data", layout="wide")
st.title("Chapter 1: Sampling, Data, and Levels of Measurement")

# ---------------------------------------------------------
# Introduction & Learning Objectives
# ---------------------------------------------------------
st.markdown("""
Welcome to the interactive module for **Chapter 1: Sampling, Data, and Levels of Measurement**. 

### The Scenario: Quantitative ETF Analysis (0050)
In this module, you are a quantitative analyst evaluating historical tick and daily flow data for the **Yuanta Taiwan 50 ETF (0050)**. Instead of memorizing abstract textbook definitions, you will interact with market data to prove statistical concepts visually.

**Your Learning Objectives:**
* **Data Classification:** Identify whether financial metrics are Nominal, Ordinal, Interval, or Ratio based on their mathematical properties.
* **Sampling Techniques:** Visually prove why drawing a purely random subset of trading days might fail to represent true market conditions, and how stratified sampling fixes this.
* **Frequency Distributions:** Summarize market volatility using relative and cumulative frequencies.
* **Statistical Rigor:** Apply textbook rounding rules to closing prices and understand experimental design in algorithmic backtesting.
""")
st.divider()

# ---------------------------------------------------------
# Data Loading & Quantitative Engineering
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Project_Alpha_Data.csv")
        
        # Rename columns to fit 0050 ETF context
        df = df.rename(columns={
            'Asset_Value': 'Closing_Price',
            'Daily_Status': 'Daily_Trend',
            'Intraday_Anomaly_Count': 'Volatility_Spikes'
        })
        
        # Map old Nominal states to Financial Nominal states
        if 'Daily_Trend' in df.columns:
            df['Daily_Trend'] = df['Daily_Trend'].replace({'Success': 'Gain', 'Failure': 'Loss'})
        
        # 1. Create Ordinal Data (Ranked risk categories)
        df['Risk_Level'] = pd.cut(
            df['Volatility_Spikes'], 
            bins=[-1, 0, 2, float('inf')], 
            labels=['Low', 'Medium', 'High']
        )
        
        # 2. Create Interval Data (Scale with no true absolute zero)
        df['Market_Heat_Index'] = np.random.normal(loc=0.0, scale=10.0, size=len(df)).round(1)
        
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Project_Alpha_Data.csv' is in the root directory.")
        st.stop()

df = load_data()

# ---------------------------------------------------------
# System Overview: The Raw Data
# ---------------------------------------------------------
st.markdown("### System Overview: The Raw Market Data")
st.markdown("Review the raw ETF logs. This represents our complete *population*.")

with st.expander("🔍 View Raw Dataset (Population)", expanded=True):
    st.dataframe(df, use_container_width=True)
    st.caption("*Note: 'Risk_Level' (Ordinal) and 'Market_Heat_Index' (Interval) have been algorithmically generated to ensure all four levels of measurement are present for analysis.*")

st.divider()

# ---------------------------------------------------------
# Section 1: Levels of Measurement (The Math Validator)
# ---------------------------------------------------------
st.markdown("### 1. Levels of Measurement Validation")
st.markdown("""
A variable's **level of measurement** dictates which mathematical operations are valid in algorithmic models. 
Test the market variables below to see how data types restrict statistical formulas.
""")

# Dynamically classify all columns in the dataset
dropdown_options = []
for col in df.columns:
    if col == 'Daily_Trend':
        level = 'Nominal'
    elif col == 'Risk_Level':
        level = 'Ordinal'
    elif col == 'Market_Heat_Index':
        level = 'Interval'
    elif pd.api.types.is_numeric_dtype(df[col]):
        level = 'Ratio'
    else:
        level = 'Nominal'
    dropdown_options.append(f"{col} ({level})")

test_var = st.selectbox("Choose a market variable to analyze:", options=dropdown_options)

# Extract the actual column name and statistical level from the dropdown string
selected_col = test_var.split(" (")[0]
selected_level = test_var.split(" (")[1].replace(")", "")

col_mode, col_sort, col_mean, col_ratio = st.columns(4)

if selected_level == "Nominal":
    with col_mode:
        top_state = df[selected_col].mode()[0]
        st.success(f"**Mode (Counting): VALID**\n\nThe most frequent value is: **{top_state}**.")
    with col_sort:
        st.error("**Sorting (Median): INVALID**\n\nDistinct categories lack an inherent mathematical sequence.")
    with col_mean:
        st.error("**Mean (Average): INVALID**\n\nYou cannot calculate the mathematical average of text labels.")
    with col_ratio:
        st.error("**Ratio (Division): INVALID**\n\nNominal data has no true zero. Multiplication is impossible.")

elif selected_level == "Ordinal":
    with col_mode:
        top_rank = df[selected_col].mode()[0]
        st.success(f"**Mode (Counting): VALID**\n\nThe most frequent rank is: **{top_rank}**.")
    with col_sort:
        st.success("**Sorting (Median): VALID**\n\nData can be logically ranked for sequential risk analysis.")
    with col_mean:
        st.error("**Mean (Average): INVALID**\n\nThe distance between ranks is not quantified. An average is mathematically unsound.")
    with col_ratio:
        st.error("**Ratio (Division): INVALID**\n\nOrdinal data lacks uniform intervals. You cannot divide 'High Risk' by 'Low Risk'.")

elif selected_level == "Interval":
    with col_mode:
        st.success("**Mode (Counting): VALID**\n\nWe can count frequent occurrences of specific index scores.")
    with col_sort:
        st.success("**Sorting (Median): VALID**\n\nIndex values naturally possess a logical order.")
    with col_mean:
        mean_val = round(df[selected_col].mean(), 1)
        st.success(f"**Mean (Average): VALID**\n\nThe distance between scores is uniform. Average is: **{mean_val}**.")
    with col_ratio:
        st.error("**Ratio (Division): INVALID**\n\nA value of 0 is a neutral baseline, not the absolute absence of activity. Proportions are invalid.")

elif selected_level == "Ratio":
    with col_mode:
        st.success("**Mode (Counting): VALID**\n\nWe can count frequent numerical occurrences.")
    with col_sort:
        st.success("**Sorting (Median): VALID**\n\nValues naturally possess a logical order.")
    with col_mean:
        mean_val = round(df[selected_col].mean(), 2)
        st.success(f"**Mean (Average): VALID**\n\nMeaningful average calculated: **{mean_val}**.")
    with col_ratio:
        st.success("**Ratio (Division): VALID**\n\nMeasurements have an absolute zero. Proportional math (like percentage returns) is perfectly valid.")

st.divider()

# ---------------------------------------------------------
# Section 2: Sampling Variations (Visual Proof)
# ---------------------------------------------------------
st.markdown("### 2. The Impact of Sampling Methods")
st.markdown("Compare how **Simple Random Sampling** can skew proportions at low sample sizes, whereas **Stratified Sampling** locks in the exact population distribution.")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    sampling_method = st.radio("Select Sampling Method:", options=["Simple Random", "Stratified"])
with col2:
    max_sample = len(df) if len(df) > 0 else 100
    sample_size = st.slider("Select Sample Size (n):", min_value=10, max_value=max_sample, value=min(25, max_sample))
    st.caption("*Try a very low sample size (e.g., n=15) to force high variance in random sampling.*")
with col3:
    st.write("") 
    st.write("") 
    st.button("🔄 Regenerate Sample")

# Execute Sampling Logic 
if sampling_method == "Simple Random":
    sampled_df = df.sample(n=sample_size, random_state=np.random.RandomState())
else:
    # Stratified Random Sampling balanced by 'Daily_Trend'
    sampled_df = df.groupby('Daily_Trend').sample(
        frac=sample_size/len(df), 
        random_state=np.random.RandomState()
    )
    if len(sampled_df) < sample_size:
        additional = df.loc[~df.index.isin(sampled_df.index)].sample(n=sample_size - len(sampled_df))
        sampled_df = pd.concat([sampled_df, additional])
    elif len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size)

# Calculate proportions and raw counts
pop_counts = df['Daily_Trend'].value_counts(normalize=True) * 100
sample_counts = sampled_df['Daily_Trend'].value_counts(normalize=True) * 100
sample_raw_counts = sampled_df['Daily_Trend'].value_counts()

# Display the True Population Baseline
status_labels = pop_counts.index.tolist()
status_values = pop_counts.values.round(1)
baseline_text = " | ".join([f"**{label}**: {val}%" for label, val in zip(status_labels, status_values)])
st.info(f"**True Population Baseline:** {baseline_text}")

# Reshape data for grouped bar chart
comparison_df = pd.DataFrame({
    'True Population %': pop_counts,
    'Sample %': sample_counts
}).fillna(0).reset_index(names='Daily_Trend')

melted_df = comparison_df.melt(id_vars='Daily_Trend', var_name='Source', value_name='Percentage')

chart_col, table_col = st.columns([3, 1])

with chart_col:
    # Side-by-side grouped bars with Y-axis locked to 100%
    chart = alt.Chart(melted_df).mark_bar().encode(
        x=alt.X('Daily_Trend:N', title="Market Trend", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Percentage:Q', title='Percentage (%)', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('Source:N', scale=alt.Scale(range=[  '#ff7f0e', '#1f77b4']), legend=alt.Legend(title="Data Source")),
        xOffset='Source:N'
    ).properties(height=350)
    
    st.altair_chart(chart, use_container_width=True)

with table_col:
    st.markdown("**Sample Draw (Raw Counts)**")
    st.dataframe(pd.DataFrame({'Count Drawn': sample_raw_counts}), use_container_width=True)
    st.caption("*Notice how drawing just 1 or 2 extra days of a specific trend causes massive percentage swings at low sample sizes.*")

st.divider()

# ---------------------------------------------------------
# Section 3: Frequency & Rounding Rules
# ---------------------------------------------------------
st.markdown("### 3. Frequency & The Rounding Rule")

freq_col, metric_col = st.columns(2)

with freq_col:
    st.markdown("**Cumulative Frequency Table**")
    st.caption("*Note: Nominal data (like 'Daily_Trend') is excluded because cumulative counting requires ordered data.*")
    
    freq_options = []
    for col in df.columns:
        if col == 'Daily_Trend':
            continue
        elif col == 'Risk_Level':
            freq_options.append(f"{col} (Ordinal)")
        elif col == 'Market_Heat_Index':
            freq_options.append(f"{col} (Continuous Interval)")
        elif pd.api.types.is_float_dtype(df[col]):
            freq_options.append(f"{col} (Continuous Ratio)")
        elif pd.api.types.is_integer_dtype(df[col]):
            freq_options.append(f"{col} (Discrete Ratio)")
        else:
            freq_options.append(f"{col} (Ordered)")
            
    freq_var = st.selectbox("Select an ordered variable to tabulate:", options=freq_options)
    selected_freq_col = freq_var.split(" (")[0]
    
    if "Continuous" in freq_var:
        st.info("Continuous data must be **binned** into ranges before counting frequencies.")
        binned_data = pd.cut(sampled_df[selected_freq_col], bins=5)
        freq_counts = binned_data.value_counts().sort_index()
    else:
        freq_counts = sampled_df[selected_freq_col].value_counts().sort_index()
        
    freq_table = pd.DataFrame({'Frequency': freq_counts})
    freq_table['Relative Frequency'] = freq_table['Frequency'] / len(sampled_df)
    freq_table['Cumulative Rel. Freq.'] = freq_table['Relative Frequency'].cumsum()
    
    st.dataframe(freq_table.style.format({
        'Relative Frequency': '{:.2f}', 
        'Cumulative Rel. Freq.': '{:.2f}'
    }), use_container_width=True)

with metric_col:
    st.markdown("**The Rounding Rule Application**")
    st.info("*Rule: Calculations must be reported to one more decimal place than the original data.*")
    
    num_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    round_var = st.selectbox("Select a numeric variable to average:", options=num_options)
    
    if not sampled_df.empty:
        raw_mean = sampled_df[round_var].mean()
        
        # Pedagogical Enforcement: Explicitly define original precision
        if round_var == 'Closing_Price':
            original_decimals = 4
        elif round_var == 'Market_Heat_Index':
            original_decimals = 1
        elif pd.api.types.is_integer_dtype(df[round_var]):
            original_decimals = 0
        else:
            original_decimals = 2 # Fallback
            
        target_decimals = original_decimals + 1
        rounded_mean = round(raw_mean, target_decimals)
        
        st.metric(
            label=f"Properly Rounded Mean ({round_var})", 
            value=f"{rounded_mean:.{target_decimals}f}", 
            delta=f"Raw calculation: {raw_mean}",
            delta_color="off"
        )

st.divider()
# ---------------------------------------------------------
# Section 4: Experimental Design & Ethics
# ---------------------------------------------------------
st.markdown("### 4. Experimental Design & Ethics")
st.warning("""
**Core Principles of Experimental Design:**
* **Randomization:** Subjects must be randomly assigned to treatments to neutralize the effect of unseen **lurking variables**.
* **Control Groups:** A baseline group receiving no treatment is strictly required to prove the explanatory variable actually caused the change.
* **Blinding:** Hiding treatment assignments from subjects (and researchers, in double-blind studies) prevents psychological bias from skewing results.

**Ethical Data Handling:**
* Cherry-picking data or manipulating sample selection to achieve a desired outcome fundamentally destroys statistical validity. 
* Transparent reporting of methodology and sample size is ethically required.
""")
