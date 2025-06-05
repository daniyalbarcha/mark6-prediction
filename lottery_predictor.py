import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests

# Set page configuration
st.set_page_config(
    page_title="Lottery Number Predictor",
    page_icon="🎲",
    layout="wide"
)

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
    </style>
    """, unsafe_allow_html=True)

class LotteryPredictor:
    def __init__(self):
        self.data = None
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def scrape_lottery_data(self, url):
        """
        Scrape lottery data from the provided URL
        This is a placeholder - you'll need to implement the actual scraping logic
        based on your specific lottery website
        """
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Implement your specific scraping logic here
            # Return DataFrame with columns: date, number1, number2, number3, number4, number5
            pass
        except Exception as e:
            st.error(f"Error scraping data: {str(e)}")
            return None

    def prepare_features(self, df):
        """Prepare features for the model"""
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_month'] = df['date'].dt.day
        return df

    def train_model(self, features, target):
        """Train the prediction model"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)

    def predict_next_numbers(self, next_date):
        """Predict lottery numbers for the next draw"""
        next_features = pd.DataFrame({
            'day_of_week': [next_date.dayofweek],
            'month': [next_date.month],
            'year': [next_date.year],
            'day_of_month': [next_date.day]
        })
        predictions = self.model.predict(next_features)
        return np.round(predictions).astype(int)

    def analyze_frequency(self, df):
        """Analyze number frequency"""
        frequencies = {}
        for i in range(1, 6):
            col = f'number{i}'
            frequencies[col] = df[col].value_counts().head(10)
        return frequencies

def main():
    st.title("🎲 Lottery Number Predictor")
    
    predictor = LotteryPredictor()
    
    # File upload section
    st.header("📊 Data Input")
    data_option = st.radio(
        "Choose data input method:",
        ("Upload CSV", "Enter Website URL")
    )
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your lottery data CSV", type=['csv'])
        if uploaded_file is not None:
            predictor.data = pd.read_csv(uploaded_file)
    else:
        url = st.text_input("Enter lottery website URL")
        if url and st.button("Fetch Data"):
            predictor.data = predictor.scrape_lottery_data(url)
    
    if predictor.data is not None:
        st.success("Data loaded successfully!")
        
        # Data Overview
        st.header("📈 Data Overview")
        st.write("Sample of loaded data:")
        st.dataframe(predictor.data.head())
        
        # Data Analysis
        st.header("📊 Historical Analysis")
        
        # Prepare data
        prepared_data = predictor.prepare_features(predictor.data.copy())
        
        # Number frequency analysis
        frequencies = predictor.analyze_frequency(predictor.data)
        
        # Display frequency plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Number Frequency Analysis")
            for num, freq in frequencies.items():
                fig = px.bar(
                    x=freq.index,
                    y=freq.values,
                    title=f"Most Common {num.capitalize()} Numbers"
                )
                st.plotly_chart(fig)
        
        with col2:
            st.subheader("Trends Over Time")
            fig = go.Figure()
            for i in range(1, 6):
                fig.add_trace(go.Scatter(
                    x=prepared_data['date'],
                    y=prepared_data[f'number{i}'],
                    name=f'Number {i}'
                ))
            fig.update_layout(title="Number Trends Over Time")
            st.plotly_chart(fig)
        
        # Prediction Section
        st.header("🎯 Number Prediction")
        
        # Prepare features for training
        features = prepared_data[['day_of_week', 'month', 'year', 'day_of_month']]
        
        # Train separate models for each number
        predictions = []
        for i in range(1, 6):
            target = prepared_data[f'number{i}']
            score = predictor.train_model(features, target)
            st.write(f"Model accuracy for number {i}: {score:.2f}")
        
        # Predict next numbers
        next_monday = datetime.now()
        while next_monday.weekday() != 0:  # 0 represents Monday
            next_monday += timedelta(days=1)
            
        st.subheader(f"Predicted Numbers for {next_monday.strftime('%Y-%m-%d')}")
        
        predicted_numbers = predictor.predict_next_numbers(next_monday)
        
        # Display predicted numbers with animation
        cols = st.columns(5)
        for i, num in enumerate(predicted_numbers):
            with cols[i]:
                st.metric(f"Number {i+1}", value=num)

if __name__ == "__main__":
    main() 