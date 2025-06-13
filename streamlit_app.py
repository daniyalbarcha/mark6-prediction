# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from lottery_scraper import LotteryScraper
from mark6_scraper import Mark6Scraper
from lottery_analyzer import LotteryAnalyzer
import numpy as np

# Page configuration
st.set_page_config(page_title="Mark 6 Predictor Pro", layout="wide", page_icon=":game_die:")

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
    num_columns = 6  # Mark 6 has 6 main numbers
    for i in range(1, num_columns + 1):
        all_numbers.extend(data[f'number{i}'].tolist())
    
    metrics['total_draws'] = len(data)
    metrics['date_range'] = (data['date'].max() - data['date'].min()).days
    metrics['most_common'] = pd.Series(all_numbers).value_counts().head(5).to_dict()
    metrics['least_common'] = pd.Series(all_numbers).value_counts().tail(5).to_dict()
    metrics['latest_date'] = data['date'].max().strftime('%Y-%m-%d')
    
    return metrics

def get_next_draw_dates(current_date=None):
    """Get the next draw dates for Mark 6 (Tue/Thu/Sat)"""
    if current_date is None:
        current_date = datetime.now()
    
    next_dates = []
    temp_date = current_date
    
    # Mark 6 draws on Tuesday, Thursday, and Saturday
    for _ in range(7):
        if temp_date.weekday() in [1, 3, 5]:  # Tuesday(1), Thursday(3), Saturday(5)
            next_dates.append(temp_date.date())
        temp_date += timedelta(days=1)
    
    return next_dates

def main():
    st.title("Mark 6 Predictor Pro")
    
    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LotteryAnalyzer()
    if 'current_predictions' not in st.session_state:
        st.session_state.current_predictions = {}
    if 'multiple_predictions' not in st.session_state:
        st.session_state.multiple_predictions = []
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    if 'total_pages' not in st.session_state:
        # Initialize scraper to get total pages
        scraper = Mark6Scraper()
        st.session_state.total_pages = scraper.get_total_pages()
    
    # Add new session state variables for prediction display
    if 'show_all_days' not in st.session_state:
        st.session_state.show_all_days = False
    if 'predicted_dates' not in st.session_state:
        st.session_state.predicted_dates = []
    
    # Initialize scraper
    scraper = Mark6Scraper()
    
    st.sidebar.header("Data Collection")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Scrape New Data", "Upload Existing Data"]
    )
    
    if data_option == "Scrape New Data":
        scrape_option = st.sidebar.radio(
            "Choose scraping method:",
            ["By Pages", "Quick Date Range"]
        )
        
        if scrape_option == "By Pages":
            st.sidebar.write("Number of pages to scrape")
            pages_to_scrape = st.sidebar.slider(
                "Pages",
                min_value=10,
                max_value=st.session_state.total_pages,
                value=min(50, st.session_state.total_pages),
                key="pages_slider"
            )
            if st.sidebar.button("Fetch Latest Data"):
                with st.spinner("Fetching historical data..."):
                    st.session_state.historical_data = scraper.scrape_historical_data(
                        start_page=1, 
                        end_page=pages_to_scrape
                    )
                    if st.session_state.historical_data is not None:
                        scraper.save_to_csv(st.session_state.historical_data)
                        st.success("Data fetched successfully!")
        else:
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.button("Last 7 Days"):
                    with st.spinner("Fetching data..."):
                        st.session_state.historical_data = scraper.scrape_by_date_range(days=7)
                        if st.session_state.historical_data is not None:
                            scraper.save_to_csv(st.session_state.historical_data)
                            st.success("Data fetched successfully!")
                
                if st.button("Last 30 Days"):
                    with st.spinner("Fetching data..."):
                        st.session_state.historical_data = scraper.scrape_by_date_range(days=30)
                        if st.session_state.historical_data is not None:
                            scraper.save_to_csv(st.session_state.historical_data)
                            st.success("Data fetched successfully!")
            
            with col2:
                if st.button("Last 90 Days"):
                    with st.spinner("Fetching data..."):
                        st.session_state.historical_data = scraper.scrape_by_date_range(days=90)
                        if st.session_state.historical_data is not None:
                            scraper.save_to_csv(st.session_state.historical_data)
                            st.success("Data fetched successfully!")
                
                if st.button("Last 365 Days"):
                    with st.spinner("Fetching data..."):
                        st.session_state.historical_data = scraper.scrape_by_date_range(days=365)
                        if st.session_state.historical_data is not None:
                            scraper.save_to_csv(st.session_state.historical_data)
                            st.success("Data fetched successfully!")
            
            # Custom date range
            st.sidebar.write("Or choose custom range:")
            days_to_fetch = st.sidebar.number_input("Number of days to look back", min_value=1, value=7)
            if st.sidebar.button("Fetch Custom Range"):
                with st.spinner("Fetching data..."):
                    st.session_state.historical_data = scraper.scrape_by_date_range(days=days_to_fetch)
                    if st.session_state.historical_data is not None:
                        scraper.save_to_csv(st.session_state.historical_data)
                        st.success("Data fetched successfully!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            st.session_state.historical_data = pd.read_csv(uploaded_file)
            st.session_state.historical_data['date'] = pd.to_datetime(st.session_state.historical_data['date'])
            st.success("Data loaded successfully!")
    
    # Main content area
    if st.session_state.historical_data is None:
        st.info("Please load or fetch data to start analysis")
        return

    # Calculate advanced metrics
    metrics = calculate_advanced_metrics(st.session_state.historical_data)
    
    # Display dataset overview
    st.header("Dataset Overview")
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
    
    # Data Visualization section
    st.subheader("Data Visualization")
    
    # Number frequency analysis
    st.write("### Number Frequency Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Number frequency chart
        all_numbers = []
        for i in range(1, 7):  # 6 main numbers for Mark 6
            all_numbers.extend(st.session_state.historical_data[f'number{i}'].tolist())
        
        number_freq = pd.Series(all_numbers).value_counts().sort_index()
        fig = px.bar(
            x=number_freq.index,
            y=number_freq.values,
            title="Number Frequency Distribution",
            labels={"x": "Number", "y": "Frequency"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display most common numbers
        st.write("Most Common Numbers:")
        st.write(pd.Series(all_numbers).value_counts().head().to_dict())
    
    with col2:
        # Least common numbers
        st.write("Least Common Numbers:")
        st.write(pd.Series(all_numbers).value_counts().tail().to_dict())
        
        # Time series of sum of numbers
        st.session_state.historical_data['numbers_sum'] = sum(
            st.session_state.historical_data[f'number{i}'] 
            for i in range(1, 7)
        )
        fig = px.line(
            st.session_state.historical_data.head(20),
            x='date',
            y='numbers_sum',
            title="Sum of Numbers Over Time (Last 20 Draws)",
            labels={"numbers_sum": "Sum of Numbers", "date": "Draw Date"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pattern Analysis
    st.write("### Pattern Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        # Even/Odd Distribution
        even_count = sum(1 for num in all_numbers if num % 2 == 0)
        odd_count = len(all_numbers) - even_count
        fig = go.Figure(data=[go.Pie(
            labels=['Even', 'Odd'],
            values=[even_count, odd_count],
            hole=.3
        )])
        fig.update_layout(title="Even/Odd Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # High/Low Distribution
        median = 25  # Mark 6 median
        high_count = sum(1 for num in all_numbers if num > median)
        low_count = len(all_numbers) - high_count
        fig = go.Figure(data=[go.Pie(
            labels=['High', 'Low'],
            values=[high_count, low_count],
            hole=.3
        )])
        fig.update_layout(title="High/Low Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sequential Analysis
    st.write("### Sequential Analysis")
    col5, col6 = st.columns(2)
    
    with col5:
        # Gap Analysis
        gaps = []
        for i in range(len(st.session_state.historical_data) - 1):
            current_numbers = set()
            next_numbers = set()
            for j in range(1, 7):
                current_numbers.add(st.session_state.historical_data.iloc[i][f'number{j}'])
                next_numbers.add(st.session_state.historical_data.iloc[i+1][f'number{j}'])
            gaps.append(len(current_numbers - next_numbers))
        
        fig = px.histogram(
            x=gaps,
            title="Number Gap Distribution",
            labels={"x": "Number of Different Numbers", "y": "Frequency"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col6:
        # Sum Distribution
        sums = st.session_state.historical_data['numbers_sum']
        fig = px.histogram(
            x=sums,
            title="Sum Distribution",
            labels={"x": "Sum of Numbers", "y": "Frequency"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Train model and make predictions
    analyzer = st.session_state.analyzer
    analyzer.train_models(st.session_state.historical_data)
    
    # Get last numbers
    last_numbers = st.session_state.historical_data.iloc[0][
        [f'number{i}' for i in range(1, 7)]
    ].values
    
    # Get next draw dates
    next_dates = get_next_draw_dates()
    
    # Prediction Controls
    st.header("Prediction Controls")
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    
    with pred_col1:
        if st.button("Predict Numbers", type="primary"):
            st.session_state.current_predictions = {}
            st.session_state.predicted_dates = next_dates
            st.session_state.show_all_days = False
            # Only predict for the first available day initially
            next_date = next_dates[0]
            st.session_state.current_predictions[next_date] = analyzer.predict_with_sequential_logic(
                last_numbers, 
                datetime.combine(next_date, datetime.min.time())
            )
            st.session_state.prediction_count += 1
    
    with pred_col2:
        if st.button("Reroll Prediction"):
            if st.session_state.current_predictions:
                if st.session_state.show_all_days:
                    # Reroll all predictions
                    st.session_state.current_predictions = {}
                    for next_date in st.session_state.predicted_dates:
                        st.session_state.current_predictions[next_date] = analyzer.reroll_prediction(
                            last_numbers,
                            datetime.combine(next_date, datetime.min.time()),
                            st.session_state.current_predictions.get(next_date)
                        )
                else:
                    # Reroll only the first day
                    next_date = st.session_state.predicted_dates[0]
                    st.session_state.current_predictions[next_date] = analyzer.reroll_prediction(
                        last_numbers,
                        datetime.combine(next_date, datetime.min.time()),
                        st.session_state.current_predictions.get(next_date)
                    )
                st.session_state.prediction_count += 1
    
    with pred_col3:
        if st.button("Show All Days"):
            st.session_state.show_all_days = True
            # Generate predictions for all available days
            for next_date in st.session_state.predicted_dates:
                if next_date not in st.session_state.current_predictions:
                    st.session_state.current_predictions[next_date] = analyzer.predict_with_sequential_logic(
                        last_numbers,
                        datetime.combine(next_date, datetime.min.time())
                    )
    
    with pred_col4:
        if st.button("Show First Day"):
            st.session_state.show_all_days = False
    
    # Display predictions
    if st.session_state.current_predictions:
        st.header("Current Predictions")
        
        # Create columns for each prediction
        num_predictions = len(st.session_state.current_predictions)
        cols = st.columns(min(num_predictions, 3))
        
        for idx, (date, prediction) in enumerate(st.session_state.current_predictions.items()):
            if not st.session_state.show_all_days and idx > 0:
                continue
                
            col_idx = idx % 3
            with cols[col_idx]:
                st.markdown(
                    f"""
                    <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;'>
                        <h3 style='color: #1f77b4; margin-bottom: 15px;'>{date.strftime('%Y-%m-%d')}</h3>
                        <h4 style='color: #2c3e50;'>Main Numbers</h4>
                        <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>
                            {"".join(f"<span style='background-color: #1f77b4; color: white; padding: 10px; border-radius: 50%; min-width: 40px; text-align: center;'>{num}</span>" for num in prediction[:-1])}
                        </div>
                        <h4 style='color: #2c3e50;'>Extra Number</h4>
                        <div style='display: flex; justify-content: center;'>
                            <span style='background-color: #e74c3c; color: white; padding: 10px; border-radius: 50%; min-width: 40px; text-align: center;'>{prediction[-1]}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # Display prediction history
    if st.session_state.multiple_predictions:
        st.header("Prediction History")
        for idx, prediction in enumerate(st.session_state.multiple_predictions):
            st.markdown(
                f"""
                <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;'>
                    <h3 style='color: #1f77b4; margin-bottom: 15px;'>Prediction {idx + 1}</h3>
                    <h4 style='color: #2c3e50;'>Main Numbers</h4>
                    <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>
                        {"".join(f"<span style='background-color: #1f77b4; color: white; padding: 10px; border-radius: 50%; min-width: 40px; text-align: center;'>{num}</span>" for num in prediction[:-1])}
                    </div>
                    <h4 style='color: #2c3e50;'>Extra Number</h4>
                    <div style='display: flex; justify-content: center;'>
                        <span style='background-color: #e74c3c; color: white; padding: 10px; border-radius: 50%; min-width: 40px; text-align: center;'>{prediction[-1]}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
