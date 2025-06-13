# Mark 6 Lottery Predictor Pro

An advanced prediction tool for Hong Kong Mark 6 lottery using machine learning and pattern analysis.

## Features

- **Data Collection**
  - Automated web scraping of historical Mark 6 results
  - Support for custom date ranges
  - CSV data export and import
  
- **Advanced Analysis**
  - Number frequency analysis
  - Pattern detection
  - Even/Odd distribution
  - High/Low number analysis
  - Sequential pattern analysis
  
- **Predictions**
  - Machine learning-based predictions
  - Support for multiple prediction methods
  - Next draw date predictions (Tue/Thu/Sat)
  - Main numbers and extra number predictions
  
- **Visualization**
  - Interactive charts and graphs
  - Number frequency distribution
  - Historical trends
  - Pattern visualization
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/daniyalbarcha/mark6-prediction.git
   cd mark6-prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser. You can:
1. Choose to scrape new data or upload existing data
2. View detailed analysis and visualizations
3. Generate predictions for upcoming draws

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- BeautifulSoup4
- NumPy
- Requests

## Data Structure

The tool uses the following data format:
- `date`: Draw date
- `number1` through `number6`: Main numbers
- `extra`: Extra/special number

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License - feel free to use this code for any purpose.

## Disclaimer

This tool is for entertainment purposes only. Lottery predictions are based on historical data analysis and cannot guarantee future results. 