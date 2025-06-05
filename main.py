import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from lottery_scraper import LotteryScraper
from lottery_analyzer import LotteryAnalyzer
import numpy as np

# Page configuration
st.set_page_config(page_title="Lottery Predictor Pro", layout="wide", page_icon="🎲")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 3rem !important;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def calculate_advanced_metrics(data):
    """Calculate advanced metrics from the dataset"""
    metrics = {}
    
    # Calculate overall statistics
    all_numbers = []
    for i in range(1, 6):
        all_numbers.extend(data[f'number{i}'].tolist())
    
    metrics['total_draws'] = len(data)
    metrics['date_range'] = (data['date'].max() - data['date'].min()).days
    metrics['most_common'] = pd.Series(all_numbers).value_counts().head(5).to_dict()
    metrics['least_common'] = pd.Series(all_numbers).value_counts().tail(5).to_dict()
    metrics['latest_date'] = data['date'].max().strftime('%Y-%m-%d')
    
    return metrics

def main():
    st.title("🎲 Lottery Predictor Pro")
    
    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LotteryAnalyzer()
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    if 'multiple_predictions' not in st.session_state:
        st.session_state.multiple_predictions = []
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    
    # Sidebar
    st.sidebar.header("Data Collection")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Scrape New Data", "Upload Existing Data"]
    )
    
    if data_option == "Scrape New Data":
        pages_to_scrape = st.sidebar.slider("Number of pages to scrape", 10, 241, 50)
        if st.sidebar.button("Fetch Latest Data"):
            with st.spinner(f"Fetching historical data from {pages_to_scrape} pages..."):
                scraper = LotteryScraper()
                st.session_state.historical_data = scraper.scrape_historical_data(
                    start_page=1, 
                    end_page=pages_to_scrape
                )
                scraper.save_to_csv(st.session_state.historical_data)
                st.success("Data fetched successfully!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            st.session_state.historical_data = pd.read_csv(uploaded_file)
            st.session_state.historical_data['date'] = pd.to_datetime(st.session_state.historical_data['date'])
            st.success("Data loaded successfully!")
    
    # Main content
    if st.session_state.historical_data is not None:
        # Calculate advanced metrics
        metrics = calculate_advanced_metrics(st.session_state.historical_data)
        
        # Display dataset overview
        st.header("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Draws Analyzed", metrics['total_draws'])
        with col2:
            st.metric("Years of Data", f"{metrics['date_range'] / 365:.1f}")
        with col3:
            st.metric("Latest Draw Date", metrics['latest_date'])
        
        # Data Overview
        st.subheader("Recent Draws")
        st.dataframe(st.session_state.historical_data.head())
        
        # Train model and make predictions
        analyzer = st.session_state.analyzer
        analyzer.train_models(st.session_state.historical_data)
        
        # Get last numbers
        last_numbers = st.session_state.historical_data.iloc[0][
            ['number1', 'number2', 'number3', 'number4', 'number5']
        ].values
        
        # Find next draw date
        next_date = datetime.now()
        while next_date.weekday() != 0:  # Find next Monday
            next_date += timedelta(days=1)
        
        # Prediction Controls
        st.header("🎯 Prediction Controls")
        pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
        
        with pred_col1:
            if st.button("🎲 Predict Numbers", type="primary"):
                st.session_state.current_prediction = analyzer.predict_with_sequential_logic(last_numbers, next_date)
                st.session_state.prediction_count += 1
        
        with pred_col2:
            if st.button("🔄 Reroll Prediction"):
                st.session_state.current_prediction = analyzer.reroll_prediction(
                    last_numbers, next_date, st.session_state.current_prediction
                )
                st.session_state.prediction_count += 1
        
        with pred_col3:
            if st.button("🎰 Predict Again"):
                st.session_state.current_prediction = analyzer.predict_with_sequential_logic(last_numbers, next_date)
                st.session_state.prediction_count += 1
        
        with pred_col4:
            prediction_count = st.selectbox("Prediction Sets", [10, 25, 50, 100], index=2)
            if st.button(f"🚀 Predict {prediction_count}x"):
                with st.spinner(f"Generating {prediction_count} prediction sets..."):
                    st.session_state.multiple_predictions = analyzer.predict_multiple_sets(
                        last_numbers, next_date, count=prediction_count, use_patterns=True
                    )
                st.success(f"Generated {prediction_count} prediction sets!")
        
        # Display Current Prediction
        if st.session_state.current_prediction:
            st.header("🎯 Current Prediction")
            st.subheader(f"Predicted Numbers for {next_date.strftime('%Y-%m-%d')} (Attempt #{st.session_state.prediction_count})")
            
            cols = st.columns(5)
            for i, num in enumerate(st.session_state.current_prediction):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div style='background-color: #e74c3c; color: white; padding: 20px; 
                        border-radius: 50%; width: 60px; height: 60px; display: flex; 
                        align-items: center; justify-content: center; font-size: 24px; margin: auto;'>
                        {num}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # Display Multiple Predictions Analysis
        if st.session_state.multiple_predictions:
            st.header("📊 Multiple Predictions Analysis")
            
            # Confidence Analysis
            confidence_analysis = analyzer.analyze_prediction_confidence(st.session_state.multiple_predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Most Confident Numbers")
                confident_numbers = confidence_analysis['most_confident'][:10]
                
                for num, confidence in confident_numbers:
                    st.progress(confidence / 100, text=f"Number {num}: {confidence:.1f}%")
            
            with col2:
                st.subheader("📈 Frequency Distribution")
                freq_data = confidence_analysis['frequency_distribution']
                fig_freq = px.bar(
                    x=list(freq_data.keys()),
                    y=list(freq_data.values()),
                    title=f"Number Frequency in {prediction_count} Predictions"
                )
                st.plotly_chart(fig_freq, use_container_width=True)
            
            # Show sample predictions
            st.subheader("🎲 Sample Prediction Sets")
            sample_size = min(10, len(st.session_state.multiple_predictions))
            
            for i in range(sample_size):
                pred_set = st.session_state.multiple_predictions[i]
                st.write(f"**Set {pred_set['set_number']}**: {pred_set['numbers']} ({pred_set['prediction_method']})")
        
        # Analysis Tabs
        st.header("📈 Detailed Analysis")
        tab1, tab2, tab3, tab4 = st.tabs(["Number Patterns", "Time Analysis", "Advanced Stats", "Sequential Patterns"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔥 Hot Numbers")
                fig_hot = px.bar(
                    x=list(metrics['most_common'].keys()),
                    y=list(metrics['most_common'].values()),
                    title="Most Frequent Numbers (All Time)"
                )
                st.plotly_chart(fig_hot, use_container_width=True)
            
            with col2:
                st.subheader("❄️ Cold Numbers")
                fig_cold = px.bar(
                    x=list(metrics['least_common'].keys()),
                    y=list(metrics['least_common'].values()),
                    title="Least Frequent Numbers (All Time)"
                )
                st.plotly_chart(fig_cold, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Number Trends Over Time")
            fig = go.Figure()
            for i in range(1, 6):
                fig.add_trace(go.Scatter(
                    x=st.session_state.historical_data['date'],
                    y=st.session_state.historical_data[f'number{i}'],
                    name=f'Number {i}'
                ))
            fig.update_layout(title="Historical Number Trends")
            st.plotly_chart(fig, use_container_width=True)
            
            # Day of week analysis
            patterns = analyzer.analyze_patterns(st.session_state.historical_data)
            st.subheader("📅 Day of Week Patterns")
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_stats = patterns['day_statistics']
            
            fig_days = go.Figure()
            for num in range(1, 6):
                values = [day_stats[f'number{num}'].get(day, 0) for day in range(7)]
                fig_days.add_trace(go.Bar(
                    name=f'Number {num}',
                    x=day_names,
                    y=values
                ))
            fig_days.update_layout(
                title="Average Numbers by Day of Week",
                barmode='group'
            )
            st.plotly_chart(fig_days, use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Number Streaks")
            # Calculate streaks
            numbers_array = np.array([st.session_state.historical_data[f'number{i}'] for i in range(1, 6)]).T
            current_streaks = {i: 0 for i in range(1, 40)}
            max_streaks = {i: 0 for i in range(1, 40)}
            
            for draw in numbers_array:
                for num in range(1, 40):
                    if num in draw:
                        current_streaks[num] += 1
                        max_streaks[num] = max(max_streaks[num], current_streaks[num])
                    else:
                        current_streaks[num] = 0
            
            streak_data = pd.DataFrame.from_dict(
                max_streaks, 
                orient='index', 
                columns=['Max Consecutive Appearances']
            )
            fig_streaks = px.bar(
                streak_data,
                title="Maximum Consecutive Appearances by Number"
            )
            st.plotly_chart(fig_streaks, use_container_width=True)
            
            # Display additional statistics
            st.subheader("📊 Additional Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Most Common Combinations")
                combinations = st.session_state.historical_data.apply(
                    lambda x: tuple(sorted([x[f'number{i}'] for i in range(1, 6)])),
                    axis=1
                ).value_counts().head(5)
                for combo, count in combinations.items():
                    st.markdown(f"**{combo}**: {count} times")
            
            with col2:
                st.markdown("### Number Distribution")
                all_nums = []
                for i in range(1, 6):
                    all_nums.extend(st.session_state.historical_data[f'number{i}'])
                fig_dist = px.histogram(
                    x=all_nums,
                    nbins=39,
                    title="Overall Number Distribution"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab4:
            st.subheader("🔍 Sequential Pattern Analysis")
            st.info("Based on your observation of +1 patterns and sequential logic in lottery draws")
            
            patterns = analyzer.analyze_patterns(st.session_state.historical_data)
            seq_patterns = patterns['sequential_patterns']
            
            # Display +1 pattern frequency
            plus_one_data = seq_patterns['plus_one_sequences']
            if plus_one_data:
                st.subheader("➕ Plus One (+1) Pattern Occurrences")
                st.write(f"Found {len(plus_one_data)} instances where numbers increased by +1 in consecutive draws")
                
                # Show recent +1 patterns
                recent_plus_one = plus_one_data[:10]
                for pattern in recent_plus_one:
                    st.write(f"**{pattern['date'].strftime('%Y-%m-%d')}**: {pattern['current']} → {pattern['next']} (+1 count: {pattern['plus_one_count']})")
            
            # Position correlations
            st.subheader("📍 Position-Based Pattern Analysis")
            pos_corr = seq_patterns['position_correlations']
            
            for pos, data in pos_corr.items():
                st.write(f"**{pos.replace('_', ' ').title()}**: Average difference = {data['mean_diff']:.2f}")
                st.write(f"Most common differences: {data['common_diffs']}")
                st.write("---")
            
            # Pattern-based prediction explanation
            st.subheader("🎯 How Pattern-Based Predictions Work")
            st.markdown("""
            Based on your analysis, the system now:
            1. **Detects +1 sequences** where numbers increment by 1 between draws
            2. **Analyzes position correlations** to find systematic changes
            3. **Applies weighted logic** favoring +1 patterns (40% weight)
            4. **Considers alternative patterns** like -1, +2 for variety
            5. **Ensures number validity** (1-39 range, no duplicates)
            """)
    
    else:
        st.info("Please load or fetch data to start analysis")

if __name__ == "__main__":
    main() 