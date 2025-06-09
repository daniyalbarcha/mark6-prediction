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

def get_next_draw_dates(current_date=None):
    """Get the next draw dates for all days except Sunday"""
    if current_date is None:
        current_date = datetime.now()
    
    next_dates = []
    temp_date = current_date
    
    # Look ahead for the next 7 days to find all draw dates
    for _ in range(7):
        if temp_date.weekday() != 6:  # 6 is Sunday
            next_dates.append(temp_date.date())
        temp_date += timedelta(days=1)
    
    return next_dates

def main():
    st.title("🎲 Lottery Predictor Pro")
    
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
        scraper = LotteryScraper()
        st.session_state.total_pages = scraper.get_total_pages()
    
    # Add new session state variables for prediction display
    if 'show_all_days' not in st.session_state:
        st.session_state.show_all_days = False
    if 'predicted_dates' not in st.session_state:
        st.session_state.predicted_dates = []
    
    # Sidebar
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
                "",
                min_value=10,
                max_value=st.session_state.total_pages,
                value=min(50, st.session_state.total_pages),
                key="pages_slider"
            )
            if st.sidebar.button("Fetch Latest Data"):
                with st.spinner(f"Fetching historical data from {pages_to_scrape} pages..."):
                    scraper = LotteryScraper()
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
                    with st.spinner("Fetching last 7 days of data..."):
                        scraper = LotteryScraper()
                        st.session_state.historical_data = scraper.scrape_by_date_range(days=7)
                        if st.session_state.historical_data is not None:
                            scraper.save_to_csv(st.session_state.historical_data)
                            st.success("Data fetched successfully!")
                
                if st.button("Last 30 Days"):
                    with st.spinner("Fetching last 30 days of data..."):
                        scraper = LotteryScraper()
                        st.session_state.historical_data = scraper.scrape_by_date_range(days=30)
                        if st.session_state.historical_data is not None:
                            scraper.save_to_csv(st.session_state.historical_data)
                            st.success("Data fetched successfully!")
            
            with col2:
                if st.button("Last 90 Days"):
                    with st.spinner("Fetching last 90 days of data..."):
                        scraper = LotteryScraper()
                        st.session_state.historical_data = scraper.scrape_by_date_range(days=90)
                        if st.session_state.historical_data is not None:
                            scraper.save_to_csv(st.session_state.historical_data)
                            st.success("Data fetched successfully!")
                
                if st.button("Last 365 Days"):
                    with st.spinner("Fetching last 365 days of data..."):
                        scraper = LotteryScraper()
                        st.session_state.historical_data = scraper.scrape_by_date_range(days=365)
                        if st.session_state.historical_data is not None:
                            scraper.save_to_csv(st.session_state.historical_data)
                            st.success("Data fetched successfully!")
            
            # Custom date range
            st.sidebar.write("Or choose custom range:")
            days_to_fetch = st.sidebar.number_input("Number of days to look back", min_value=1, value=7)
            if st.sidebar.button("Fetch Custom Range"):
                with st.spinner(f"Fetching last {days_to_fetch} days of data..."):
                    scraper = LotteryScraper()
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
    
    # Get next draw dates
    next_dates = get_next_draw_dates()
    
    # Prediction Controls
    st.header("🎯 Prediction Controls")
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    
    with pred_col1:
        if st.button("🎲 Predict Numbers", type="primary"):
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
        if st.button("🔄 Reroll Prediction"):
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
        if st.button("🎰 Predict Again"):
            st.session_state.current_predictions = {}
            if st.session_state.show_all_days:
                # Predict all days
                for next_date in st.session_state.predicted_dates:
                    st.session_state.current_predictions[next_date] = analyzer.predict_with_sequential_logic(
                        last_numbers,
                        datetime.combine(next_date, datetime.min.time())
                    )
            else:
                # Predict only first day
                next_date = st.session_state.predicted_dates[0]
                st.session_state.current_predictions[next_date] = analyzer.predict_with_sequential_logic(
                    last_numbers,
                    datetime.combine(next_date, datetime.min.time())
                )
            st.session_state.prediction_count += 1
    
    with pred_col4:
        prediction_count = st.selectbox("Prediction Sets", [10, 25, 50, 100], index=2)
        if st.button(f"🚀 Predict {prediction_count}x"):
            with st.spinner(f"Generating {prediction_count} prediction sets..."):
                st.session_state.multiple_predictions = []
                dates_to_predict = st.session_state.predicted_dates if st.session_state.show_all_days else [st.session_state.predicted_dates[0]]
                for next_date in dates_to_predict:
                    predictions = analyzer.predict_multiple_sets(
                        last_numbers,
                        datetime.combine(next_date, datetime.min.time()),
                        count=prediction_count,
                        use_patterns=True
                    )
                    for pred in predictions:
                        pred['date'] = next_date
                    st.session_state.multiple_predictions.extend(predictions)
            st.success(f"Generated {prediction_count} prediction sets!")

    # Display Current Predictions
    if st.session_state.current_predictions:
        st.header("🎯 Current Predictions")
        
        # Show first day prediction
        next_date = st.session_state.predicted_dates[0]
        prediction = st.session_state.current_predictions[next_date]
        st.subheader(f"Predicted Numbers for {next_date.strftime('%Y-%m-%d')} ({next_date.strftime('%A')}) - Attempt #{st.session_state.prediction_count}")
        
        cols = st.columns(5)
        for i, num in enumerate(prediction):
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
        
        # Add button to show/hide remaining days
        if len(st.session_state.predicted_dates) > 1:
            if not st.session_state.show_all_days:
                if st.button("🔮 Predict Next 5 Days"):
                    st.session_state.show_all_days = True
                    # Predict remaining days
                    for next_date in st.session_state.predicted_dates[1:]:
                        st.session_state.current_predictions[next_date] = analyzer.predict_with_sequential_logic(
                            last_numbers,
                            datetime.combine(next_date, datetime.min.time())
                        )
            
            # Show remaining days if requested
            if st.session_state.show_all_days:
                st.markdown("---")
                st.subheader("Predictions for Remaining Days:")
                for next_date in st.session_state.predicted_dates[1:]:
                    if next_date in st.session_state.current_predictions:
                        prediction = st.session_state.current_predictions[next_date]
                        st.write(f"**{next_date.strftime('%Y-%m-%d')} ({next_date.strftime('%A')})**")
                        
                        cols = st.columns(5)
                        for i, num in enumerate(prediction):
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
                        st.markdown("---")

    # Display Multiple Predictions Analysis
    if st.session_state.multiple_predictions:
        st.header("📊 Multiple Predictions Analysis")
        
        # Group predictions by date
        predictions_by_date = {}
        for pred in st.session_state.multiple_predictions:
            date = pred['date']
            if date not in predictions_by_date:
                predictions_by_date[date] = []
            predictions_by_date[date].append(pred)
        
        for date, predictions in predictions_by_date.items():
            st.subheader(f"Analysis for {date.strftime('%Y-%m-%d')} ({date.strftime('%A')})")
            
            # Confidence Analysis for this date
            confidence_analysis = analyzer.analyze_prediction_confidence(predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("🎯 Most Confident Numbers")
                confident_numbers = confidence_analysis['most_confident'][:10]
                
                for num, confidence in confident_numbers:
                    st.progress(confidence / 100, text=f"Number {num}: {confidence:.1f}%")
            
            with col2:
                st.write("📈 Frequency Distribution")
                freq_data = confidence_analysis['frequency_distribution']
                fig_freq = px.bar(
                    x=list(freq_data.keys()),
                    y=list(freq_data.values()),
                    title=f"Number Frequency in {len(predictions)} Predictions"
                )
                st.plotly_chart(fig_freq, use_container_width=True)
            
            # Show sample predictions for this date
            st.write("🎲 Sample Prediction Sets")
            sample_size = min(5, len(predictions))
            
            for i in range(sample_size):
                pred_set = predictions[i]
                st.write(f"**Set {pred_set['set_number']}**: {pred_set['numbers']} ({pred_set['prediction_method']})")
            
            st.markdown("---")

    # Analysis Tabs
    st.header("📈 Detailed Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Number Patterns", "Time Analysis", "Advanced Stats", "Sequential Patterns"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 Hot Numbers")
            st.write("Most frequently drawn numbers:")
            
            # Calculate frequency for each possible number
            all_numbers = []
            for i in range(1, 6):
                all_numbers.extend(st.session_state.historical_data[f'number{i}'].tolist())
            
            number_freq = pd.Series(all_numbers).value_counts()
            total_draws = len(st.session_state.historical_data)
            
            # Calculate frequency percentage
            number_freq_pct = (number_freq / total_draws * 100).round(1)
            
            # Create bar chart for hot numbers
            hot_df = pd.DataFrame({
                'Number': number_freq.head(10).index,
                'Frequency': number_freq.head(10).values
            }).sort_values('Frequency', ascending=True)  # Ascending for better visualization
            
            fig_hot = go.Figure(go.Bar(
                x=hot_df['Frequency'],
                y=hot_df['Number'].astype(str),
                orientation='h',
                text=hot_df['Frequency'],
                textposition='auto',
                marker_color='orangered'
            ))
            
            fig_hot.update_layout(
                title="Top 10 Most Frequent Numbers",
                xaxis_title="Number of Times Drawn",
                yaxis_title="Number",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_hot, use_container_width=True)
            
            # Display number grid
            st.write("\n**Number Grid**")
            grid_cols = st.columns(8)
            for i in range(39):
                col_idx = i % 8
                with grid_cols[col_idx]:
                    num = i + 1
                    count = number_freq.get(num, 0)
                    freq = (count / total_draws * 100)
                    
                    # Determine color based on frequency
                    if freq >= number_freq_pct.quantile(0.75):
                        bg_color = "rgba(255,69,0,0.2)"  # Hot
                    elif freq <= number_freq_pct.quantile(0.25):
                        bg_color = "rgba(135,206,250,0.2)"  # Cold
                    else:
                        bg_color = "rgba(128,128,128,0.1)"  # Neutral
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 5px; border-radius: 5px; background-color: {bg_color}; margin-bottom: 5px;'>
                        <div style='font-size: 1.2em; font-weight: bold;'>{num}</div>
                        <div style='font-size: 0.9em;'>{freq:.1f}%</div>
                        <div style='font-size: 0.8em;'>({count})</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("❄️ Cold Numbers")
            st.write("Least frequently drawn numbers:")
            
            # Create bar chart for cold numbers
            cold_df = pd.DataFrame({
                'Number': number_freq.tail(10).index,
                'Frequency': number_freq.tail(10).values
            }).sort_values('Frequency', ascending=False)  # Descending for better visualization
            
            fig_cold = go.Figure(go.Bar(
                x=cold_df['Frequency'],
                y=cold_df['Number'].astype(str),
                orientation='h',
                text=cold_df['Frequency'],
                textposition='auto',
                marker_color='deepskyblue'
            ))
            
            fig_cold.update_layout(
                title="Top 10 Least Frequent Numbers",
                xaxis_title="Number of Times Drawn",
                yaxis_title="Number",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_cold, use_container_width=True)
            
            # Overdue numbers analysis
            st.write("\n**📅 Overdue Numbers**")
            current_date = st.session_state.historical_data['date'].max()
            
            overdue_numbers = {}
            for num in range(1, 40):
                last_drawn = None
                for i in range(1, 6):
                    num_dates = st.session_state.historical_data[
                        st.session_state.historical_data[f'number{i}'] == num
                    ]['date']
                    if not num_dates.empty:
                        last_date = num_dates.max()
                        if last_drawn is None or last_date > last_drawn:
                            last_drawn = last_date
                
                if last_drawn is not None:
                    days_since = (current_date - last_drawn).days
                    overdue_numbers[num] = days_since
            
            # Create bar chart for overdue numbers
            overdue_sorted = dict(sorted(overdue_numbers.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig_overdue = go.Figure(go.Bar(
                x=list(overdue_sorted.values()),
                y=[str(k) for k in overdue_sorted.keys()],
                orientation='h',
                text=list(overdue_sorted.values()),
                textposition='auto',
                marker_color='mediumseagreen'
            ))
            
            fig_overdue.update_layout(
                title="Most Overdue Numbers",
                xaxis_title="Days Since Last Drawn",
                yaxis_title="Number",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_overdue, use_container_width=True)

    with tab2:
        st.subheader("📊 Time-based Analysis")
        
        # Date range selector
        date_range = st.date_input(
            "Select Date Range",
            value=(
                st.session_state.historical_data['date'].min().date(),
                st.session_state.historical_data['date'].max().date()
            ),
            min_value=st.session_state.historical_data['date'].min().date(),
            max_value=st.session_state.historical_data['date'].max().date()
        )
        
        # Filter data based on date range
        mask = (
            st.session_state.historical_data['date'].dt.date >= date_range[0]
        ) & (
            st.session_state.historical_data['date'].dt.date <= date_range[1]
        )
        filtered_data = st.session_state.historical_data[mask]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week analysis
            st.write("**Day of Week Analysis**")
            
            day_stats = {}
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for day in range(7):
                day_data = filtered_data[filtered_data['date'].dt.weekday == day]
                if not day_data.empty:
                    day_stats[days[day]] = {
                        'avg': [day_data[f'number{i}'].mean() for i in range(1, 6)],
                        'count': len(day_data)
                    }
            
            # Create day of week visualization
            day_df = pd.DataFrame({
                'Day': list(day_stats.keys()),
                'Average': [sum(stats['avg'])/5 for stats in day_stats.values()],
                'Count': [stats['count'] for stats in day_stats.values()]
            })
            
            fig_days = go.Figure(data=[
                go.Bar(name='Draw Count', x=day_df['Day'], y=day_df['Count']),
                go.Scatter(name='Average Number', x=day_df['Day'], y=day_df['Average'], yaxis='y2')
            ])
            
            fig_days.update_layout(
                title="Draws by Day of Week",
                yaxis=dict(title="Number of Draws"),
                yaxis2=dict(title="Average Number", overlaying='y', side='right'),
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig_days, use_container_width=True)
        
        with col2:
            # Monthly analysis
            st.write("**Monthly Analysis**")
            
            monthly_stats = {}
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            
            for month in range(1, 13):
                month_data = filtered_data[filtered_data['date'].dt.month == month]
                if not month_data.empty:
                    monthly_stats[months[month-1]] = {
                        'avg': [month_data[f'number{i}'].mean() for i in range(1, 6)],
                        'count': len(month_data)
                    }
            
            # Create monthly visualization
            month_df = pd.DataFrame({
                'Month': list(monthly_stats.keys()),
                'Average': [sum(stats['avg'])/5 for stats in monthly_stats.values()],
                'Count': [stats['count'] for stats in monthly_stats.values()]
            })
            
            fig_months = go.Figure(data=[
                go.Bar(name='Draw Count', x=month_df['Month'], y=month_df['Count']),
                go.Scatter(name='Average Number', x=month_df['Month'], y=month_df['Average'], yaxis='y2')
            ])
            
            fig_months.update_layout(
                title="Draws by Month",
                yaxis=dict(title="Number of Draws"),
                yaxis2=dict(title="Average Number", overlaying='y', side='right'),
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig_months, use_container_width=True)

    with tab3:
        st.subheader("📊 Advanced Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Position Analysis**")
            
            # Create position-based visualization
            pos_stats = []
            for pos in range(1, 6):
                numbers = st.session_state.historical_data[f'number{pos}']
                pos_stats.append({
                    'Position': f'Position {pos}',
                    'Mean': numbers.mean(),
                    'Median': numbers.median(),
                    'Std': numbers.std(),
                    'Most Common': numbers.mode().iloc[0],
                    'Most Common Count': numbers.value_counts().iloc[0]
                })
            
            pos_df = pd.DataFrame(pos_stats)
            
            fig_pos = go.Figure()
            fig_pos.add_trace(go.Bar(
                name='Mean',
                x=pos_df['Position'],
                y=pos_df['Mean'],
                text=pos_df['Mean'].round(1),
                textposition='auto'
            ))
            fig_pos.add_trace(go.Bar(
                name='Most Common',
                x=pos_df['Position'],
                y=pos_df['Most Common'],
                text=pos_df['Most Common'],
                textposition='auto'
            ))
            
            fig_pos.update_layout(
                title="Number Statistics by Position",
                yaxis_title="Number",
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig_pos, use_container_width=True)
            
            # Even/Odd Analysis
            st.write("\n**Even/Odd Distribution**")
            even_counts = st.session_state.historical_data.apply(
                lambda row: sum(1 for i in range(1, 6) if row[f'number{i}'] % 2 == 0),
                axis=1
            )
            
            even_dist = even_counts.value_counts().sort_index()
            fig_even_odd = go.Figure(go.Bar(
                x=[f"{i} even, {5-i} odd" for i in even_dist.index],
                y=even_dist.values,
                text=even_dist.values,
                textposition='auto'
            ))
            
            fig_even_odd.update_layout(
                title="Even/Odd Number Distribution",
                xaxis_title="Distribution",
                yaxis_title="Number of Draws",
                height=400
            )
            st.plotly_chart(fig_even_odd, use_container_width=True)
        
        with col2:
            st.write("**Number Combinations**")
            
            # Sum analysis
            number_columns = [f'number{i}' for i in range(1, 6)]
            sums = st.session_state.historical_data[number_columns].sum(axis=1)
            
            fig_sums = go.Figure(go.Histogram(
                x=sums,
                nbinsx=30,
                name="Sum Distribution"
            ))
            
            fig_sums.update_layout(
                title="Distribution of Number Sums",
                xaxis_title="Sum of Numbers",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_sums, use_container_width=True)
            
            # Consecutive numbers analysis
            st.write("\n**Consecutive Numbers Analysis**")
            consecutive_counts = st.session_state.historical_data.apply(
                lambda row: sum(1 for i in range(len(number_columns)-1)
                              if row[number_columns[i+1]] - row[number_columns[i]] == 1),
                axis=1
            )
            
            consec_dist = consecutive_counts.value_counts().sort_index()
            fig_consec = go.Figure(go.Bar(
                x=[f"{i} pairs" for i in consec_dist.index],
                y=consec_dist.values,
                text=consec_dist.values,
                textposition='auto'
            ))
            
            fig_consec.update_layout(
                title="Consecutive Number Pairs Distribution",
                xaxis_title="Number of Consecutive Pairs",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_consec, use_container_width=True)

    with tab4:
        st.subheader("🔄 Sequential Pattern Analysis")
        
        # Get sequential patterns from analyzer
        patterns = st.session_state.analyzer.analyze_patterns(st.session_state.historical_data)
        seq_patterns = patterns.get('sequential_patterns', {})
        
        # Display sequential patterns
        st.write("**Position-based Sequential Patterns**")
        
        # Create a more detailed sequential pattern visualization
        for pos in range(1, 6):
            st.write(f"Position {pos}:")
            
            # Get position correlations
            pos_corr = seq_patterns.get('position_correlations', {}).get(str(pos), {})
            
            if pos_corr:
                mean_diff = pos_corr.get('mean_diff', 0)
                common_diffs = pos_corr.get('common_diffs', {})
                
                # Convert common_diffs to a DataFrame for better visualization
                diff_data = pd.DataFrame({
                    'Difference': list(map(int, common_diffs.keys())),
                    'Frequency': list(common_diffs.values())
                }).sort_values('Frequency', ascending=False).head(5)
                
                # Display statistics
                st.write(f"- Average difference between consecutive numbers: {mean_diff:.2f}")
                st.write("- Most common differences:")
                
                # Create bar chart for differences
                fig_diff = px.bar(
                    diff_data,
                    x='Difference',
                    y='Frequency',
                    title=f"Most Common Differences - Position {pos}",
                    labels={
                        'Difference': 'Number Difference',
                        'Frequency': 'Occurrences'
                    }
                )
                fig_diff.update_layout(
                    showlegend=False,
                    xaxis=dict(tickmode='linear', dtick=1),
                    yaxis=dict(rangemode='nonnegative')
                )
                st.plotly_chart(fig_diff, use_container_width=True)
            else:
                st.write("No significant patterns found")
            
            st.write("---")
        
        # Display overall sequence patterns
        st.write("**Overall Sequence Analysis**")
        
        # Calculate overall sequence statistics
        all_diffs = []
        for i in range(len(st.session_state.historical_data)):
            numbers = sorted([
                st.session_state.historical_data.iloc[i][f'number{j}']
                for j in range(1, 6)
            ])
            diffs = [numbers[j] - numbers[j-1] for j in range(1, len(numbers))]
            all_diffs.extend(diffs)
        
        # Create DataFrame for overall differences
        overall_diff_data = pd.DataFrame({
            'Difference': all_diffs
        })
        
        # Create histogram for overall differences
        fig_overall = px.histogram(
            overall_diff_data,
            x='Difference',
            nbins=20,
            title="Distribution of Differences Between Consecutive Numbers",
            labels={
                'Difference': 'Number Difference',
                'count': 'Frequency'
            }
        )
        fig_overall.update_layout(
            showlegend=False,
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_overall, use_container_width=True)
        
        # Display summary statistics
        st.write("**Summary Statistics**")
        st.write(f"- Average gap between numbers: {np.mean(all_diffs):.2f}")
        st.write(f"- Most common gap: {pd.Series(all_diffs).mode().iloc[0]}")
        st.write(f"- Minimum gap: {min(all_diffs)}")
        st.write(f"- Maximum gap: {max(all_diffs)}")

if __name__ == "__main__":
    main() 