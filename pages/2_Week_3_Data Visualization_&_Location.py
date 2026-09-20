import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Week 2: Visuals & Location", page_icon="📊", layout="wide")

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('project_alpha.csv', parse_dates=['Date'])
        df = df.rename(columns={'Asset_Value': 'Closing_Price', 'Daily_Status': 'Daily_Trend', 'Intraday_Anomaly_Count': 'Volatility_Spikes'})
    except FileNotFoundError:
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100)
        prices = np.random.normal(100, 15, 100)
        prices[95:] = prices[95:] + 50 
        trends = np.random.choice(['Gain', 'Loss'], 100, p=[0.55, 0.45])
        spikes = np.random.poisson(2, 100)
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
st.title("📊 Intuition Engine: Data Visualization & Location")

st.success("""
**🎯 Core Objective:** Statistics is just storytelling with numbers. 
Use the interactive tools below to see how changing parameters alters the mathematical story. 
*Do not just look at the graphs—play with them!*
""")

with st.expander("🔍 Step 1: Inspect the Raw Data (Overview)", expanded=False):
    st.markdown("Before we graph anything, look at the raw data. This is **Project Alpha**.")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(8), use_container_width=True)
    with col2:
        st.markdown("""
        **Variables:**
        * **Date:** When the measurement was taken.
        * **Closing_Price:** Quantitative (Continuous). Real values to 2 decimals.
        * **Daily_Trend:** Categorical (Nominal). 'Gain' or 'Loss'.
        * **Volatility_Spikes:** Quantitative (Discrete). Daily count of anomalies.
        """)

main_tab1, main_tab2 = st.tabs(["📉 Part 1: Displaying Data", "🎯 Part 2: Measures of Location"])

# ==========================================
# PART 1: DISPLAYING DATA
# ==========================================
with main_tab1:
    t_stem, t_time, t_bar, t_hist, t_lie = st.tabs([
        "1.1 Stem-and-Leaf", 
        "1.2 & 1.6 Time Series", 
        "1.3 Bar Graphs", 
        "1.4 & 1.5 Histograms", 
        "1.7 How NOT to Lie"
    ])
    # --- 1.1 Stem-and-Leaf ---
    with t_stem:
        st.info("💡 **Concept:** A stem-and-leaf plot splits numbers into a 'stem' (leading digits) and 'leaf' (final digit).")
        
        #stem_tab1, stem_tab2 = st.tabs(["Example 1: Decimals (Prices)", "Example 2: Integers (Small Sample)"])
        stem_tab1, stem_tab2 = st.tabs(["Example 1: Integers (Small Sample)", "Example 2: Decimals (Prices)"])
        
        with stem_tab1:
            c3, c4 = st.columns([1, 2])
            with c3:
                st.markdown("### 📖 Example 1: Integers (N=30)")
                st.markdown("""
                To understand the core mechanic, let's look at a smaller sample of 30 random two-digit integers (e.g., Daily Trade Volume).
                * **Stem:** The 'Tens' digit.
                * **Leaf:** The 'Units' digit.
                
                **Example:** `12 | 1` represents **121**
                
                *(Note: Single digits like `7` have a stem of `0`, so they appear as `0 | 7`)*
                """)
                
            with c4:
                # Generate a clean toy dataset of 30 integers
                np.random.seed(123)
                int_vals = np.sort(np.random.randint(55, 125, size=30))
                
                stem_dict2 = {}
                for v in int_vals:
                    stem, leaf = divmod(int(v), 10)
                    stem_dict2.setdefault(stem, []).append(str(leaf))
                
                result2 = "Stem (Tens) | Leaves (Units)\n"
                result2 += "-" * 35 + "\n"
                for s in sorted(stem_dict2.keys()):
                    leaves = " ".join(stem_dict2[s])
                    result2 += f"{s:11} | {leaves}\n"
                st.code(result2, language="text")    
        with stem_tab2:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("### 📖 Example 2: Decimals (N=100)")
                st.markdown("""
                Because `Closing_Price` has two decimal places (e.g., $102.45), we must define our units to make the plot readable.
                * **Stem:** The whole dollar amount (Integer).
                * **Leaf:** The *dimes* column (First decimal / Tenths). 
                * *(We truncate the pennies/hundredths for clarity).*
                
                **Example:** `102 | 4` represents **$102.4x**
                
                *(Note: We randomly sampled 100 data points from our 1000-day dataset so the plot remains readable on your screen!)*
                """)
                
            with c2:
                # We pull a random sample of 100 from the full dataset
                vals = df['Closing_Price'].dropna().sample(n=100, random_state=42).sort_values()
                stem_dict = {}
                for v in vals:
                    stem = int(v)
                    leaf = int((v * 10) % 10)
                    stem_dict.setdefault(stem, []).append(str(leaf))
                
                result = "Stem (Dollars) | Leaves (First Decimal / 0.1)\n"
                result += "-" * 35 + "\n"
                for s in sorted(stem_dict.keys()):
                    leaves = " ".join(stem_dict[s])
                    result += f"{s:14} | {leaves}\n"
                st.code(result, language="text")   

    # --- 1.2 & 1.6 Time Series ---
    with t_time:
        st.info("💡 **Concept:** Time series graphs plot data chronologically to spot trends, seasonality, or structural breaks over time.")
        smooth = st.checkbox("Apply 7-Day Moving Average (Trend Line)")
        
        fig = px.line(df, x='Date', y='Closing_Price', markers=True)
        if smooth:
            df['7D_MA'] = df['Closing_Price'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['7D_MA'], mode='lines', name='7-Day Trend', line=dict(color='red', width=3)))
        st.plotly_chart(fig, use_container_width=True)

    # --- 1.3 Bar Graphs (Updated to Counts vs Aggregations) ---
    with t_bar:
        st.info("💡 **Concept:** Bar graphs are used for Categorical data." \
        "The bars DO NOT touch because categories (like 'Gain' and 'Loss') are separate, discrete groups. " )
        st.info("They can show the **Frequency** (how many days were Gains vs. Losses) OR they can show an **Aggregated Value** (like the total number of volatility spikes that occurred on those days).")
        
        # Calculate Frequency Data
        freq_data = df['Daily_Trend'].value_counts().reset_index()
        freq_data.columns = ['Daily_Trend', 'Count']
        
        # Calculate Aggregated Data (Sum of Spikes by Trend)
        spike_data = df.groupby('Daily_Trend')['Volatility_Spikes'].sum().reset_index()
        spike_data.columns = ['Daily_Trend', 'Total_Spikes']
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Frequency (Counts)")
            st.markdown("*How many days were Gains vs. Losses?*")
            fig_freq = px.bar(freq_data, x='Daily_Trend', y='Count', text='Count')
            fig_freq.update_layout(bargap=0.4) 
            st.plotly_chart(fig_freq, use_container_width=True)
            
        with c2:
            st.markdown("### Total Volatility Spikes")
            st.markdown("*Which trend experienced more total volatility spikes?*")
            fig_spikes = px.bar(spike_data, x='Daily_Trend', y='Total_Spikes', text='Total_Spikes')
            fig_spikes.update_layout(bargap=0.4, yaxis_title="Total Spikes") 
            st.plotly_chart(fig_spikes, use_container_width=True)

    # # --- 1.3 Bar Graphs ---
    # with t_bar:
    #     st.info("💡 **Concept:** Bar graphs are used for Categorical data. Notice how the **shape** is identical whether we look at raw counts or percentages.")
        
    #     bar_data = df['Daily_Trend'].value_counts().reset_index()
    #     bar_data.columns = ['Daily_Trend', 'Count']
    #     bar_data['Percentage'] = (bar_data['Count'] / bar_data['Count'].sum()) * 100
        
    #     c1, c2 = st.columns(2)
    #     with c1:
    #         st.markdown("### Frequency (Counts)")
    #         fig_freq = px.bar(bar_data, x='Daily_Trend', y='Count', text='Count')
    #         fig_freq.update_layout(bargap=0.4) 
    #         st.plotly_chart(fig_freq, use_container_width=True)
    #     with c2:
    #         st.markdown("### Relative Frequency (Percentages)")
    #         fig_rel = px.bar(bar_data, x='Daily_Trend', y='Percentage', text='Percentage')
    #         fig_rel.update_traces(texttemplate='%{text:.1f}%')
    #         fig_rel.update_layout(bargap=0.4, yaxis_title="Percentage (%)") 
    #         st.plotly_chart(fig_rel, use_container_width=True)

    # --- 1.4 & 1.5 Histograms & Polygons (With Relative Frequency) ---
    with t_hist:
        st.info("💡 **Concept:** Histograms group continuous numbers into 'Bins'. The bars TOUCH because the number line is continuous.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            num_bins = st.slider("Number of Bins (Buckets):", min_value=3, max_value=25, value=10)
            hist_y_axis = st.radio("Y-Axis Metric:", ["Frequency (Count)", "Relative Frequency (%)"])
            show_polygon = st.checkbox("Overlay Frequency Polygon", value=True)
            
            # Strict NumPy calculations to force exact bins
            raw_counts, bin_edges = np.histogram(df['Closing_Price'].dropna(), bins=num_bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            
            total_data_points = len(df['Closing_Price'].dropna())
            relative_freqs = (raw_counts / total_data_points) * 100
            
            st.markdown("### 🪣 Current Bins")
            st.markdown(f"Total Data Points (N): **{total_data_points}**")
            
            bin_df = pd.DataFrame({
                "Range": [f"${bin_edges[i]:.0f} -${bin_edges[i+1]:.0f}" for i in range(len(bin_edges)-1)],
                "Freq": raw_counts,
                "Rel Freq": [f"{v:.1f}%" for v in relative_freqs]
            })
            st.dataframe(bin_df, hide_index=True)
            
        with col2:
            fig_hist = go.Figure()
            
            # Switch Y-axis data based on toggle
            y_data = relative_freqs if hist_y_axis == "Relative Frequency (%)" else raw_counts
            y_title = "Relative Frequency (%)" if hist_y_axis == "Relative Frequency (%)" else "Frequency (Count)"
            hover_fmt = "%{y:.1f}%" if hist_y_axis == "Relative Frequency (%)" else "%{y}"
            
            # Base Histogram mapped perfectly to NumPy edges
            fig_hist.add_trace(go.Bar(
                x=bin_centers, y=y_data, width=bin_widths, 
                name='Histogram', marker_color='#636EFA', 
                marker_line_color='black', marker_line_width=1.5,
                hovertemplate=hover_fmt
            ))
            
            # Frequency Polygon
            if show_polygon:
                poly_x = np.insert(bin_centers, 0, bin_centers[0] - bin_widths[0])
                poly_x = np.append(poly_x, bin_centers[-1] + bin_widths[-1])
                poly_y = np.insert(y_data, 0, 0)
                poly_y = np.append(poly_y, 0)
                
                fig_hist.add_trace(go.Scatter(
                    x=poly_x, y=poly_y, mode='lines+markers', 
                    name='Polygon', line=dict(color='red', width=3),
                    hovertemplate=hover_fmt
                ))
            
            # Force X-axis to display exact bin boundaries
            fig_hist.update_layout(
                bargap=0, 
                xaxis=dict(tickvals=bin_edges, tickformat=".0f", title="Closing Price"),
                yaxis=dict(title=y_title)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- 1.7 How NOT to Lie (Requirement 5) ---
    with t_lie:
        st.error("⚠️ **Visual Distortion:** Graphs can be mathematically accurate but visually deceptive. As a FinTech professional, you must spot these tricks.")
        
        st.markdown("### Trick 1: The Y-Axis Zoom (Exaggerating Volatility)")
        y_min_slider = st.slider("Adjust Y-Axis Minimum Limit:", min_value=0.0, max_value=float(df['Closing_Price'].min()), value=float(df['Closing_Price'].min())/2)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**A: The 'Distorted' View (Looks highly volatile)**")
            fig_lie1 = px.line(df, x='Date', y='Closing_Price').update_yaxes(range=[y_min_slider, df['Closing_Price'].max()])
            st.plotly_chart(fig_lie1, use_container_width=True)
        with c2:
            st.markdown("**B: The 'Honest' View (Starts at zero)**")
            fig_lie2 = px.line(df, x='Date', y='Closing_Price').update_yaxes(range=[0, df['Closing_Price'].max() * 1.1])
            st.plotly_chart(fig_lie2, use_container_width=True)
            
        st.markdown("---")

        st.markdown("### Trick 2: Cherry-Picking Timeframes (Hiding the Trend)")
        st.markdown("By shrinking our window, we can make a stable asset look highly volatile, or turn a long-term bull market into a 'crash'.")
        
        max_idx = len(df) - 15
        window_start = st.slider("Select the start day for a 15-day window:", 0, max_idx, 5)
        
        zoom_df = df.iloc[window_start : window_start + 15]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**A: The Cherry-Picked View (The 'Story')**")
            fig_zoom = px.line(zoom_df, x='Date', y='Closing_Price', line_shape='linear')
            fig_zoom.update_traces(line_color='red')
            st.plotly_chart(fig_zoom, use_container_width=True)
            
        with c2:
            st.markdown("**B: The Full Context (The Truth)**")
            fig_full = px.line(df, x='Date', y='Closing_Price')
            # Highlight the exact cherry-picked area
            fig_full.add_vrect(
                x0=zoom_df['Date'].min(), x1=zoom_df['Date'].max(), 
                fillcolor="red", opacity=0.3, line_width=0, annotation_text="Hidden Window"
            )
            st.plotly_chart(fig_full, use_container_width=True)

# ==========================================
# PART 2: MEASURES OF LOCATION
# ==========================================
with main_tab2:
    st.info("💡 **Concept:** Before using complex formulas, understand that location measures simply rely on **putting data in order from smallest to largest**.")
    
    # Generate the base N=11 sample used across the page
    np.random.seed(99)
    small_sample = np.sort(np.random.choice(df['Closing_Price'], size=11, replace=False))
    n_small = len(small_sample)
    
    # st.markdown("### 🔢 Our Base Small Sample (N = 11)")
    # st.markdown("We drew 11 random closing prices from our dataset and **sorted them from smallest to largest**.")
    
    # cols = st.columns(11)
    # for i, val in enumerate(small_sample):
    #     bg_color, border, label = "#f0f2f6", "1px solid #ccc", f"Pos {i+1}"
    #     if i == 2:   
    #         bg_color, border, label = "#d4edda", "2px solid #28a745", "Q1 (Pos 3)"
    #     elif i == 5: 
    #         bg_color, border, label = "#cce5ff", "2px solid #007bff", "Med (Pos 6)"
    #     elif i == 8: 
    #         bg_color, border, label = "#fff3cd", "2px solid #ffc107", "Q3 (Pos 9)"
            
    #     cols[i].markdown(f"<div style='text-align:center; background-color:{bg_color}; padding:8px; border-radius:5px; border:{border}; font-size:14px;'><b>{label}</b><br>{val:.1f}</div>", unsafe_allow_html=True)
    
    st.write("")
    loc_t1, loc_t2 = st.tabs(["2.1 Intuition of Quartiles", "2.2 Intuition of Percentiles"])
    
    # --- 2.1 Quartiles Intuition (Odd vs Even) ---
    # --- 2.1 Quartiles & Outliers Intuition ---
    with loc_t1:
        st.subheader("Finding Quartiles and Outliers Step-by-Step")
        
        st.markdown("#### Scenario A: Odd Samples (N=11)")
        st.markdown("We drew 11 random closing prices from our dataset and **sorted them from smallest to largest**.")
        
        cols = st.columns(11)
        for i, val in enumerate(small_sample):
            bg_color, border, label = "#f0f2f6", "1px solid #ccc", f"Pos {i+1}"
            if i == 2:   
                bg_color, border, label = "#d4edda", "2px solid #28a745", "Q1 (Pos 3)"
            elif i == 5: 
                bg_color, border, label = "#cce5ff", "2px solid #007bff", "Med (Pos 6)"
            elif i == 8: 
                bg_color, border, label = "#fff3cd", "2px solid #ffc107", "Q3 (Pos 9)"
                
            cols[i].markdown(f"<div style='text-align:center; background-color:{bg_color}; padding:8px; border-radius:5px; border:{border}; font-size:14px;'><b>{label}</b><br>{val:.1f}</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1.2])
        
        q1_a = small_sample[2]
        med_a = small_sample[5]
        q3_a = small_sample[8]
        iqr_a = q3_a - q1_a
        lower_fence_a = q1_a - (1.5 * iqr_a)
        upper_fence_a = q3_a + (1.5 * iqr_a)
        outliers_a = [v for v in small_sample if v < lower_fence_a or v > upper_fence_a]
        outlier_txt_a = f"**Yes!** {outliers_a} falls outside the fences." if outliers_a else "**None.** All data points are inside the fences."
        
        with c1:
            st.markdown(f"""
            **Step 1: Find the Median (Q2)**
            * The median cuts the data exactly in half. For N=11, the middle is uniquely **Position 6**. 
            * **<span style='color:#007bff;'>Q2 = {med_a:.2f}</span>**
            
            **Step 2: Find Q1 (Median of the lower half)**
            * Look at values *below* Q2 (Positions 1 to 5). The middle of those 5 numbers is Position 3.
            * **<span style='color:#28a745;'>Q1 = {q1_a:.2f}</span>**
            
            **Step 3: Find Q3 (Median of the upper half)**
            * Look at values *above* Q2 (Positions 7 to 11). The middle is Position 9.
            * **<span style='color:#ffc107;'>Q3 = {q3_a:.2f}</span>**
            
            **Step 4: Interquartile Range (IQR)**
            * *Intuition: The IQR measures the spread of the middle 50% of the data.*
            * IQR = Q3 - Q1 = {q3_a:.2f} - {q1_a:.2f} = **{iqr_a:.2f}**
            
            **Step 5: Identify Outliers (1.5x IQR Rule)**
            * *Intuition: Any value too far from the middle 50% is flagged as an outlier.*
            * Lower Fence = Q1 - 1.5(IQR) = **{lower_fence_a:.2f}**
            * Upper Fence = Q3 + 1.5(IQR) = **{upper_fence_a:.2f}**
            * *Are there outliers?* {outlier_txt_a}
            """, unsafe_allow_html=True)
            
        with c2:
            fig_box = px.box(x=small_sample, points="all", title="Horizontal Box Plot Mapping (N=11)", orientation="h")
            fig_box.update_traces(quartilemethod="exclusive")
            fig_box.update_layout(xaxis_title="Closing Price", yaxis_title="", height=350)
            fig_box.add_vline(x=q1_a, line_dash="dot", line_color="#28a745", annotation_text="Q1")
            fig_box.add_vline(x=med_a, line_dash="dot", line_color="#007bff", annotation_text="Median")
            fig_box.add_vline(x=q3_a, line_dash="dot", line_color="#ffc107", annotation_text="Q3")
            # Add fences
            fig_box.add_vline(x=lower_fence_a, line_dash="dash", line_color="red", annotation_text="Lower Fence")
            fig_box.add_vline(x=upper_fence_a, line_dash="dash", line_color="red", annotation_text="Upper Fence")
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        
        st.markdown("#### Scenario B: Even Samples (N=10)")
        st.markdown("Let's draw a new sample of **10 items** to see how the math changes when there is no single middle number.")
        
        np.random.seed(101)
        even_sample = np.sort(np.random.choice(df['Closing_Price'], size=10, replace=False))
        
        cols_even = st.columns(10)
        for i, val in enumerate(even_sample):
            bg_color, border, label = "#f0f2f6", "1px solid #ccc", f"Pos {i+1}"
            if i == 2:   
                bg_color, border, label = "#d4edda", "2px solid #28a745", "Q1 (Pos 3)"
            elif i == 4 or i == 5: 
                bg_color, border, label = "#cce5ff", "2px solid #007bff", f"Med Base"
            elif i == 7: 
                bg_color, border, label = "#fff3cd", "2px solid #ffc107", "Q3 (Pos 8)"
                
            cols_even[i].markdown(f"<div style='text-align:center; background-color:{bg_color}; padding:8px; border-radius:5px; border:{border}; font-size:14px;'><b>{label}</b><br>{val:.1f}</div>", unsafe_allow_html=True)
        
        c3, c4 = st.columns([1, 1.2])
        
        q1_b = even_sample[2]
        med_b = (even_sample[4] + even_sample[5]) / 2
        q3_b = even_sample[7]
        iqr_b = q3_b - q1_b
        lower_fence_b = q1_b - (1.5 * iqr_b)
        upper_fence_b = q3_b + (1.5 * iqr_b)
        outliers_b = [v for v in even_sample if v < lower_fence_b or v > upper_fence_b]
        outliers_str_b = ", ".join([f"{v:.2f}" for v in outliers_b])
        outlier_txt_b = f"**Yes!** {outliers_str_b} falls outside the fences." if outliers_b else "**None.** All data points are inside the fences."
        
        with c3:
            st.markdown(f"""
            **Step 1: Find the Median (Q2)**
            * For N=10, there is no single middle number. It falls *between* **Position 5 and 6**. 
            * We average them: `({even_sample[4]:.2f} + {even_sample[5]:.2f}) / 2`
            * **<span style='color:#007bff;'>Q2 = {med_b:.2f}</span>**
            
            **Step 2: Find Q1 (Median of the lower half)**
            * The lower half is Positions 1 to 5. The middle of these 5 numbers is Position 3.
            * **<span style='color:#28a745;'>Q1 = {q1_b:.2f}</span>**
            
            **Step 3: Find Q3 (Median of the upper half)**
            * The upper half is Positions 6 to 10. The middle of these 5 numbers is Position 8.
            * **<span style='color:#ffc107;'>Q3 = {q3_b:.2f}</span>**
            
            **Step 4: Interquartile Range (IQR)**
            * IQR = Q3 - Q1 = {q3_b:.2f} - {q1_b:.2f} = **{iqr_b:.2f}**
            
            **Step 5: Identify Outliers (1.5x IQR Rule)**
            * Lower Fence = Q1 - 1.5(IQR) = **{lower_fence_b:.2f}**
            * Upper Fence = Q3 + 1.5(IQR) = **{upper_fence_b:.2f}**
            * *Are there outliers?* {outlier_txt_b}
            """, unsafe_allow_html=True)
            
        with c4:
            fig_box_even = px.box(x=even_sample, points="all", title="Horizontal Box Plot Mapping (N=10)", orientation="h")
            fig_box_even.update_traces(quartilemethod="exclusive")
            fig_box_even.update_layout(xaxis_title="Closing Price", yaxis_title="", height=350)
            fig_box_even.add_vline(x=q1_b, line_dash="dot", line_color="#28a745", annotation_text="Q1")
            fig_box_even.add_vline(x=med_b, line_dash="dot", line_color="#007bff", annotation_text="Median")
            fig_box_even.add_vline(x=q3_b, line_dash="dot", line_color="#ffc107", annotation_text="Q3")
            # Add fences
            fig_box_even.add_vline(x=lower_fence_b, line_dash="dash", line_color="red", annotation_text="Lower Fence")
            fig_box_even.add_vline(x=upper_fence_b, line_dash="dash", line_color="red", annotation_text="Upper Fence")
            st.plotly_chart(fig_box_even, use_container_width=True)
    # # --- 2.1 Quartiles Intuition (Odd vs Even directly following each other) ---
    # with loc_t1:
    #     st.subheader("Finding Quartiles Step-by-Step")
        
    #     st.markdown("#### Scenario A: Odd Samples (N=11)")
    #     st.markdown("We drew 11 random closing prices from our dataset and **sorted them from smallest to largest**.")
        
    #     cols = st.columns(11)
    #     for i, val in enumerate(small_sample):
    #         bg_color, border, label = "#f0f2f6", "1px solid #ccc", f"Pos {i+1}"
    #         if i == 2:   
    #             bg_color, border, label = "#d4edda", "2px solid #28a745", "Q1 (Pos 3)"
    #         elif i == 5: 
    #             bg_color, border, label = "#cce5ff", "2px solid #007bff", "Med (Pos 6)"
    #         elif i == 8: 
    #             bg_color, border, label = "#fff3cd", "2px solid #ffc107", "Q3 (Pos 9)"
                
    #         cols[i].markdown(f"<div style='text-align:center; background-color:{bg_color}; padding:8px; border-radius:5px; border:{border}; font-size:14px;'><b>{label}</b><br>{val:.1f}</div>", unsafe_allow_html=True)
        
    #     c1, c2 = st.columns([1, 1.2])
    #     with c1:
    #         st.markdown(f"""
    #         **Step 1: Find the Median (Q2)**
    #         * The median cuts the data exactly in half. For N=11, the middle is uniquely **Position 6**. 
    #         * **<span style='color:#007bff;'>Q2 = {small_sample[5]:.2f}</span>**
            
    #         **Step 2: Find Q1 (Median of the lower half)**
    #         * Look at values *below* Q2 (Positions 1 to 5). The middle of those 5 numbers is Position 3.
    #         * **<span style='color:#28a745;'>Q1 = {small_sample[2]:.2f}</span>**
            
    #         **Step 3: Find Q3 (Median of the upper half)**
    #         * Look at values *above* Q2 (Positions 7 to 11). The middle is Position 9.
    #         * **<span style='color:#ffc107;'>Q3 = {small_sample[8]:.2f}</span>**
    #         """, unsafe_allow_html=True)
            
    #     with c2:
    #         fig_box = px.box(x=small_sample, points="all", title="Horizontal Box Plot Mapping (N=11)", orientation="h")
    #         fig_box.update_layout(xaxis_title="Closing Price", yaxis_title="", height=250)
    #         fig_box.add_vline(x=small_sample[2], line_dash="dot", line_color="#28a745", annotation_text="Q1")
    #         fig_box.add_vline(x=small_sample[5], line_dash="dot", line_color="#007bff", annotation_text="Median")
    #         fig_box.add_vline(x=small_sample[8], line_dash="dot", line_color="#ffc107", annotation_text="Q3")
    #         st.plotly_chart(fig_box, use_container_width=True)

    #     st.markdown("---")
        
    #     st.markdown("#### Scenario B: Even Samples (N=10)")
    #     st.markdown("Let's draw a new sample of **10 items** to see how the math changes when there is no single middle number.")
        
    #     np.random.seed(101)
    #     even_sample = np.sort(np.random.choice(df['Closing_Price'], size=10, replace=False))
        
    #     cols_even = st.columns(10)
    #     for i, val in enumerate(even_sample):
    #         bg_color, border, label = "#f0f2f6", "1px solid #ccc", f"Pos {i+1}"
    #         if i == 2:   
    #             bg_color, border, label = "#d4edda", "2px solid #28a745", "Q1 (Pos 3)"
    #         elif i == 4 or i == 5: 
    #             bg_color, border, label = "#cce5ff", "2px solid #007bff", f"Med Base"
    #         elif i == 7: 
    #             bg_color, border, label = "#fff3cd", "2px solid #ffc107", "Q3 (Pos 8)"
                
    #         cols_even[i].markdown(f"<div style='text-align:center; background-color:{bg_color}; padding:8px; border-radius:5px; border:{border}; font-size:14px;'><b>{label}</b><br>{val:.1f}</div>", unsafe_allow_html=True)
        
    #     c3, c4 = st.columns([1, 1.2])
    #     med_even = (even_sample[4] + even_sample[5]) / 2
        
    #     with c3:
    #         st.markdown(f"""
    #         **Step 1: Find the Median (Q2)**
    #         * For N=10, there is no single middle number. It falls *between* **Position 5 and 6**. 
    #         * We average them: `({even_sample[4]:.2f} + {even_sample[5]:.2f}) / 2`
    #         * **<span style='color:#007bff;'>Q2 = {med_even:.2f}</span>**
            
    #         **Step 2: Find Q1 (Median of the lower half)**
    #         * The lower half is Positions 1 to 5. The middle of these 5 numbers is Position 3.
    #         * **<span style='color:#28a745;'>Q1 = {even_sample[2]:.2f}</span>**
            
    #         **Step 3: Find Q3 (Median of the upper half)**
    #         * The upper half is Positions 6 to 10. The middle of these 5 numbers is Position 8.
    #         * **<span style='color:#ffc107;'>Q3 = {even_sample[7]:.2f}</span>**
    #         """, unsafe_allow_html=True)
            
    #     with c4:
    #         fig_box_even = px.box(x=even_sample, points="all", title="Horizontal Box Plot Mapping (N=10)", orientation="h")
    #         fig_box_even.update_layout(xaxis_title="Closing Price", yaxis_title="", height=250)
    #         fig_box_even.add_vline(x=even_sample[2], line_dash="dot", line_color="#28a745", annotation_text="Q1")
    #         fig_box_even.add_vline(x=med_even, line_dash="dot", line_color="#007bff", annotation_text="Median")
    #         fig_box_even.add_vline(x=even_sample[7], line_dash="dot", line_color="#ffc107", annotation_text="Q3")
    #         st.plotly_chart(fig_box_even, use_container_width=True)



    # --- 2.2 Percentiles Intuition (Fully Updated) ---

    with loc_t2:
        st.markdown("### 🔢 Our Base Small Sample (N = 11)")
        st.markdown("We drew 11 random closing prices from our dataset and **sorted them from smallest to largest**.")
                
        cols = st.columns(11)
        for i, val in enumerate(small_sample):
            bg_color, border, label = "#f0f2f6", "2px solid #ccc", f"Pos {i+1}"
            # if i == 2:   
            #     bg_color, border, label = "#d4edda", "2px solid #28a745", "Q1 (Pos 3)"
            # elif i == 5: 
            #     bg_color, border, label = "#cce5ff", "2px solid #007bff", "Med (Pos 6)"
            # elif i == 8: 
            #     bg_color, border, label = "#fff3cd", "2px solid #ffc107", "Q3 (Pos 9)"
                     
            cols[i].markdown(f"<div style='text-align:center; background-color:{bg_color}; padding:8px; border-radius:5px; border:{border}; font-size:14px;'><b>{label}</b><br>{val:.1f}</div>", unsafe_allow_html=True)

         
        perc_tab1, perc_tab2 = st.tabs(["Method 1: Find the Value from a Percentile", "Method 2: Find the Percentile of a Value"])
   
        # --- Method 1: k-th Percentile Formula ---
        with perc_tab1:
            st.subheader("Formula for Finding the $k$th Percentile")
            c1, c2 = st.columns([1, 1.5])
            
            with c1:
                k_percentile = st.slider("Select k (Percentile):", 1, 99, 25)
                index_i = (k_percentile / 100) * (n_small + 1)
                
                st.markdown(f"""
                **Step 1: Calculate the Index ($i$)**  
                $i = \\frac{{k}}{{100}}(n + 1)$  
                $i = \\frac{{{k_percentile}}}{{100}}({n_small} + 1) = \\mathbf{{{index_i:.2f}}}$
                """)
                
            with c2:
                if index_i.is_integer():
                    pos = int(index_i)
                    st.success(f"""
                    **Step 2: Evaluate Index**  
                    If $i$ is an integer, the $k$th percentile is the data value in the $i$th position.  
                    
                    Since **{pos}** is an integer, the **{k_percentile}th percentile** is the value at Position {pos}: **{small_sample[pos-1]:.2f}**.
                    """)
                else:
                    lower_pos = math.floor(index_i)
                    upper_pos = math.ceil(index_i)
                    val_lower = small_sample[lower_pos-1]
                    val_upper = small_sample[upper_pos-1]
                    avg_val = (val_lower + val_upper) / 2
                    
                    st.warning(f"""
                    **Step 2: Evaluate Index (Averaging)**  
                    If $i$ is not an integer, round $i$ up and round $i$ down to the nearest integers, then average the two data values in these positions.
                    
                    1. Round down **{index_i:.2f}** $\\rightarrow$ Position **{lower_pos}** (Value: {val_lower:.2f})  
                    2. Round up **{index_i:.2f}** $\\rightarrow$ Position **{upper_pos}** (Value: {val_upper:.2f})  
                    
                    **Average:** $\\frac{{{val_lower:.2f} + {val_upper:.2f}}}{{2}} = \\mathbf{{{avg_val:.2f}}}$
                    
                    The **{k_percentile}th percentile** is **{avg_val:.2f}**.
                    """)
                    
        # --- Method 2: Percentile of a Value Formula ---
        with perc_tab2:
            st.subheader("Formula for Finding the Percentile of a Value")
            c3, c4 = st.columns([1, 1.5])
            
            with c3:
                target_val = st.selectbox("Select a value from our sample to evaluate:", small_sample)
                
                x_count = sum(1 for v in small_sample if v < target_val)
                y_count = sum(1 for v in small_sample if v == target_val)
                
                st.markdown(f"""
                **Count the Variables:**
                * **$x$** (Count of values *strictly less than* {target_val:.2f}) = **{x_count}**
                * **$y$** (Count of values *equal to* {target_val:.2f}) = **{y_count}**
                * **$n$** (Total number of data points) = **{n_small}**
                """)
                
            with c4:
                perc_calc = ((x_count + 0.5 * y_count) / n_small) * 100
                perc_rounded = round(perc_calc)
                
                st.info(f"""
                **Step 1: Calculate the Raw Percentile**  
                $\\text{{Percentile}} = \\frac{{x + 0.5y}}{{n}}(100)$  
                $\\text{{Percentile}} = \\frac{{{x_count} + 0.5({y_count})}}{{{n_small}}}(100) = \\mathbf{{{perc_calc:.2f}}}$
                
                **Step 2: Round to the Nearest Integer**  
                Rounding {perc_calc:.2f} gives us **{perc_rounded}**.
                
                **Conclusion:**  
                The value **{target_val:.2f}** falls at the **{perc_rounded}th percentile** of this dataset.
                """)