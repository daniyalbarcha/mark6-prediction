# Mark 6 Lottery Predictor Pro

An advanced prediction tool for Hong Kong Mark 6 lottery using machine learning and AI models (Claude, GPT-4, and Grok).

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
  
- **AI-Powered Predictions**
  - Multiple AI model support:
    - Claude (Anthropic)
    - GPT-4 (OpenAI)
    - Grok
  - Traditional ML predictions
  - Cost-effective caching system
  - Confidence scoring
  - Detailed explanations
  
- **Visualization**
  - Interactive charts and graphs
  - Number frequency distribution
  - Historical trends
  - Pattern visualization
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mark6-prediction.git
   cd mark6-prediction
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up API keys:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - CLAUDE_API_KEY from [Anthropic](https://www.anthropic.com/)
     - OPENAI_API_KEY from [OpenAI](https://platform.openai.com/)
     - GROK_API_KEY from Grok's platform

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Using AI Predictions:
   - Toggle "Use AI for Prediction"
   - Select your preferred AI model:
     - Claude (~$0.50 per prediction)
     - GPT-4 (~$0.80 per prediction)
     - Grok (~$0.30 per prediction)
   - Click "Generate Prediction"

3. Prediction Controls:
   - "Reroll Prediction": Generate new numbers using the same method
   - "Show All Days": Display predictions for all upcoming draw dates
   - "Show First Day": Display prediction for next draw only

4. Bulk Analysis:
   - Run multiple predictions to identify patterns
   - View top 8 most frequent numbers
   - Analyze numbers 9-20 for additional insights

## Cost Management

The app includes several features to manage API costs:

1. **Caching System**
   - Predictions are cached for 1 hour
   - Reusing cached predictions incurs no additional cost
   - Cache can be cleared manually if needed

2. **Cost Tracking**
   - Real-time cost display per prediction
   - Session total cost tracking
   - Cost estimates before generating predictions

3. **Model Selection**
   - Choose models based on cost-effectiveness
   - Option to use traditional ML (no cost)
   - Configurable model parameters in .env file

## Requirements

- Python 3.8+
- API keys for chosen AI models
- Internet connection for web scraping
- Sufficient API credits

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

This tool is for entertainment purposes only. Lottery predictions are based on historical data analysis and AI models but cannot guarantee future results. Use responsibly and be aware of API costs when using AI features. 