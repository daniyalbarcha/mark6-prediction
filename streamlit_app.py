# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from lottery_scraper import LotteryScraper
from mark6_scraper import Mark6Scraper
from lottery_analyzer import LotteryAnalyzer
from context_engineering import ContextManager, BulkPredictor, GrokAPI, OpenAIAPI
import numpy as np
from config import *
import os
from dotenv import load_dotenv
import anthropic

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
    .cost-warning {
        color: #dc3545;
        font-weight: bold;
    }
    .ai-toggle {
        padding: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def handle_data_upload():
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data['date'] = pd.to_datetime(data['date'])
            st.success("Data loaded successfully!")
            return data
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
    return None

def handle_data_scraping(days):
    try:
        scraper = Mark6Scraper()
        data = scraper.scrape_by_date_range(days=days)
        if data is not None:
            scraper.save_to_csv(data)
            st.success("Data fetched successfully!")
            return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
    return None

def save_api_keys(claude_key, grok_key, openai_key):
    """Save API keys to .env file"""
    try:
        # Create .env file if it doesn't exist
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        
        # Read existing content to preserve other variables
        existing_content = {}
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        existing_content[key] = value

        # Update with new values if provided
        if claude_key:
            existing_content['CLAUDE_API_KEY'] = claude_key
        if grok_key:
            existing_content['GROK_API_KEY'] = grok_key
        if openai_key:
            existing_content['OPENAI_API_KEY'] = openai_key

        # Write back to file
        with open(env_path, 'w') as f:
            for key, value in existing_content.items():
                f.write(f"{key}={value}\n")

        # Reload environment variables
        load_dotenv(env_path, override=True)
        
        # Update the clients with new API keys
        if 'context_manager' in st.session_state:
            if claude_key:
                st.session_state.context_manager.claude_client = anthropic.Anthropic(api_key=claude_key)
            if grok_key:
                st.session_state.context_manager.grok_client = GrokAPI(api_key=grok_key)
            if openai_key:
                st.session_state.context_manager.openai_client = OpenAIAPI(api_key=openai_key)

        return True
    except Exception as e:
        print(f"Error saving API keys: {str(e)}")
        return False

def load_api_keys():
    """Load API keys from .env file"""
    try:
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
        return {
            'claude': os.getenv('CLAUDE_API_KEY', ''),
            'grok': os.getenv('GROK_API_KEY', ''),
            'openai': os.getenv('OPENAI_API_KEY', '')
        }
    except Exception as e:
        print(f"Error loading API keys: {str(e)}")
        return {'claude': '', 'grok': '', 'openai': ''}

def settings_page():
    st.title("⚙️ Settings")
    
    # Load current API keys
    current_keys = load_api_keys()
    
    st.header("🔑 API Keys Configuration")
    
    with st.form("api_keys_form"):
        # Claude API Key
        claude_key = st.text_input(
            "Claude API Key",
            value=current_keys['claude'],
            type="password",
            help="Enter your Claude API key from Anthropic"
        )
        
        # GPT-4 API Key
        openai_key = st.text_input(
            "OpenAI API Key",
            value=current_keys['openai'],
            type="password",
            help="Enter your OpenAI API key for GPT-4"
        )
        
        # Grok API Key
        grok_key = st.text_input(
            "Grok API Key",
            value=current_keys['grok'],
            type="password",
            help="Enter your Grok API key"
        )
        
        submitted = st.form_submit_button("Save API Keys", type="primary")
        if submitted:
            if save_api_keys(claude_key, grok_key, openai_key):
                st.success("✅ API keys saved successfully!")
                st.info("Changes will take effect immediately.")
            else:
                st.error("Failed to save API keys. Please check file permissions.")

def main():
    st.title("Mark 6 Predictor Pro")
    
    # Add page navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Settings"]
    )
    
    if page == "Settings":
        settings_page()
        return
    
    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LotteryAnalyzer()
    if 'context_manager' not in st.session_state:
        st.session_state.context_manager = ContextManager()
    if 'bulk_predictor' not in st.session_state:
        st.session_state.bulk_predictor = BulkPredictor(
            st.session_state.analyzer,
            st.session_state.context_manager
        )
    if 'total_cost' not in st.session_state:
        st.session_state.total_cost = 0.0
    if 'total_pages' not in st.session_state:
        try:
            scraper = Mark6Scraper()
            st.session_state.total_pages = scraper.get_total_pages()
        except Exception as e:
            print(f"Warning: Failed to get total pages - {str(e)}")
            st.session_state.total_pages = 141  # Default max pages
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False

    # Data Collection Section
    st.sidebar.header("Data Collection")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Scrape New Data", "Upload Existing Data"]
    )

    if data_source == "Scrape New Data":
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
                value=50,
                key="pages_slider"
            )
            if st.sidebar.button("Fetch Latest Data", key="fetch_pages_button"):
                with st.spinner("Fetching historical data..."):
                    try:
                        scraper = Mark6Scraper()
                        st.session_state.historical_data = scraper.scrape_historical_data(
                            start_page=1, 
                            end_page=pages_to_scrape
                        )
                        if st.session_state.historical_data is not None:
                            scraper.save_to_csv(st.session_state.historical_data)
                            st.success("Data fetched successfully!")
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")

        else:  # Quick Date Range
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("Last 7 Days", key="7days_button"):
                    with st.spinner("Fetching data..."):
                        st.session_state.historical_data = handle_data_scraping(7)
                if st.button("Last 30 Days", key="30days_button"):
                    with st.spinner("Fetching data..."):
                        st.session_state.historical_data = handle_data_scraping(30)
            with col2:
                if st.button("Last 90 Days", key="90days_button"):
                    with st.spinner("Fetching data..."):
                        st.session_state.historical_data = handle_data_scraping(90)
                if st.button("Last 365 Days", key="365days_button"):
                    with st.spinner("Fetching data..."):
                        st.session_state.historical_data = handle_data_scraping(365)

            # Custom date range
            st.sidebar.write("Or choose custom range:")
            days_to_fetch = st.sidebar.number_input(
                "Number of days to look back",
                min_value=1,
                value=7,
                key="custom_days"
            )
            if st.sidebar.button("Fetch Custom Range", key="custom_range_button"):
                with st.spinner("Fetching data..."):
                    st.session_state.historical_data = handle_data_scraping(days_to_fetch)

    else:  # Upload Existing Data
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")
        if uploaded_file is not None:
            try:
                st.session_state.historical_data = pd.read_csv(uploaded_file)
                st.session_state.historical_data['date'] = pd.to_datetime(st.session_state.historical_data['date'])
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")

    # After data is loaded, train models if needed
    if st.session_state.historical_data is not None and not st.session_state.models_trained:
        with st.spinner("Training prediction models..."):
            try:
                st.session_state.analyzer.train_models(st.session_state.historical_data)
                st.session_state.models_trained = True
                st.success("Models trained successfully!")
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
                return

    # Main content area
    if st.session_state.historical_data is None:
        st.info("Please load or fetch data to start analysis")
        return

    # Display data overview
    st.header("📊 Data Overview")
    st.write("Recent Draws:")
    st.dataframe(st.session_state.historical_data.head())

    # Display basic statistics
    st.header("📈 Basic Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Draws", len(st.session_state.historical_data))
    with col2:
        date_range = (st.session_state.historical_data['date'].max() - 
                     st.session_state.historical_data['date'].min()).days
        st.metric("Date Range (days)", date_range)
    with col3:
        st.metric("Latest Draw", st.session_state.historical_data['date'].max().strftime('%Y-%m-%d'))

    # Data Visualization section
    st.header("📊 Data Analysis")
    
    # Number frequency analysis
    st.subheader("Number Frequency Analysis")
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
    st.subheader("Pattern Analysis")
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

    # Prediction Controls Section
    st.header("🎯 Number Prediction")
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        single_ai_enabled = st.toggle("Use AI for Prediction", value=False, key="single_ai_toggle")
    
    selected_model = "none"
    if single_ai_enabled:
        with pred_col2:
            model_options = {
                "claude": "Claude (Cost: ~$0.50/prediction)",
                "gpt-4": "GPT-4 (Cost: ~$0.80/prediction)",
                "grok": "Grok (Cost: ~$0.30/prediction)"
            }
            selected_model = st.selectbox(
                "Select AI Model",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                key="single_prediction_model"
            )
            
            cost_warnings = {
                "claude": "Using Claude will cost approximately $0.50 per prediction",
                "gpt-4": "Using GPT-4 will cost approximately $0.80 per prediction",
                "grok": "Using Grok will cost approximately $0.30 per prediction"
            }
            st.warning(f"⚠️ {cost_warnings[selected_model]}")
        
        with pred_col3:
            st.info(f"💰 Total cost this session: ${st.session_state.total_cost:.2f}")

    # Generate Prediction Button
    if st.button("Generate Prediction", type="primary", key="single_predict_button"):
        with st.spinner("Generating prediction..."):
            try:
                target_date = datetime.now()
                
                if single_ai_enabled:
                    # Get AI prediction
                    context = st.session_state.context_manager.create_context(
                        historical_data=st.session_state.historical_data,
                        target_date=target_date,
                        ai_enabled=True
                    )
                    response = st.session_state.context_manager.get_ai_prediction(model=selected_model)
                    
                    main_numbers = response.prediction[:6]
                    extra_number = response.prediction[6] if len(response.prediction) > 6 else 0
                    
                    st.session_state.total_cost += response.cost
                    
                    # Store prediction
                    st.session_state.prediction_history.append({
                        'date': target_date,
                        'main_numbers': main_numbers,
                        'extra_number': extra_number,
                        'ai_enabled': True,
                        'model': selected_model,
                        'cost': response.cost,
                        'explanation': response.explanation,
                        'confidence': response.confidence_score
                    })
                    
                    # Display prediction
                    st.success("🤖 AI-Generated Prediction")
                    
                    # Display numbers
                    main_cols = st.columns(6)
                    for i, num in enumerate(main_numbers):
                        with main_cols[i]:
                            st.metric(f"Number {i+1}", value=int(num))
                    st.metric("Extra Number", value=int(extra_number), delta="Special")
                    
                    # Show details
                    st.info(f"💡 AI Explanation: {response.explanation}")
                    st.progress(response.confidence_score, text=f"Confidence Score: {response.confidence_score:.2%}")
                    st.info(f"💰 Cost for this prediction: ${response.cost:.2f}")
                
                else:
                    # Use traditional ML prediction
                    last_numbers = st.session_state.historical_data.iloc[0][
                        ['number1', 'number2', 'number3', 'number4', 'number5', 'number6', 'extra']
                    ].values
                    main_numbers, extra_number = st.session_state.analyzer.predict_next_numbers(
                        last_numbers,
                        target_date
                    )
                    
                    # Store prediction
                    st.session_state.prediction_history.append({
                        'date': target_date,
                        'main_numbers': main_numbers,
                        'extra_number': extra_number,
                        'ai_enabled': False,
                        'model': 'ML',
                        'cost': 0
                    })
                    
                    # Display prediction
                    st.success("🔮 ML-Generated Prediction")
                    main_cols = st.columns(6)
                    for i, num in enumerate(main_numbers):
                        with main_cols[i]:
                            st.metric(f"Number {i+1}", value=int(num))
                    st.metric("Extra Number", value=int(extra_number), delta="Special")
            
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
                st.error("Please try again or contact support if the issue persists.")

    # Show prediction history
    if st.session_state.prediction_history:
        st.header("📜 Previous Predictions")
        for idx, pred in enumerate(reversed(st.session_state.prediction_history[-5:])):
            with st.expander(f"Prediction {len(st.session_state.prediction_history)-idx} - {pred['date'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"Main Numbers: {pred['main_numbers']}")
                st.write(f"Extra Number: {pred['extra_number']}")
                st.write(f"Method: {'AI (' + pred['model'] + ')' if pred['ai_enabled'] else 'Traditional ML'}")
                if pred['ai_enabled']:
                    st.write(f"Confidence: {pred.get('confidence', 0):.2%}")
                    st.write(f"Explanation: {pred.get('explanation', 'N/A')}")
                    st.write(f"Cost: ${pred['cost']:.2f}")

    # Bulk Analysis Section
    st.header("📊 Bulk Analysis")
    bulk_col1, bulk_col2 = st.columns(2)
    
    with bulk_col1:
        iterations = st.number_input(
            "Number of predictions to analyze",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            key="bulk_iterations"
        )
        
        if st.button("Run Analysis", type="primary", key="bulk_analyze_button"):
            with st.spinner("Analyzing patterns..."):
                target_date = datetime.now()
                bulk_results = st.session_state.bulk_predictor.predict_bulk(
                    historical_data=st.session_state.historical_data,
                    target_date=target_date,
                    iterations=iterations,
                    ai_enabled=single_ai_enabled,
                    model=selected_model if single_ai_enabled else "none"
                )
                
                if single_ai_enabled:
                    st.session_state.total_cost += bulk_results['total_cost']
                
                # Display results
                st.subheader("🎯 Top 8 Most Frequent Numbers")
                st.write(bulk_results['top_8_numbers'])
                
                st.subheader("📊 Numbers 9-20")
                st.write(bulk_results['next_12_numbers'])
                
                # Frequency visualization
                freq_df = pd.DataFrame(
                    list(bulk_results['number_frequencies'].items()),
                    columns=['Number', 'Frequency']
                )
                fig = px.bar(
                    freq_df.head(20),
                    x='Number',
                    y='Frequency',
                    title="Top 20 Number Frequencies"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if single_ai_enabled:
                    st.info(f"💰 Cost for this analysis: ${bulk_results['total_cost']:.2f}")
                    st.info(f"💰 Average cost per prediction: ${bulk_results['average_cost_per_prediction']:.3f}")

if __name__ == "__main__":
    main()
