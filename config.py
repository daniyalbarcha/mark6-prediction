import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AI Configuration
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
GROK_API_KEY = os.getenv('GROK_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Cost Configuration
CLAUDE_COST_PER_TOKEN = 0.000008  # $0.008 per 1K tokens
GROK_COST_PER_TOKEN = 0.000005   # Example cost, adjust based on actual pricing
OPENAI_INPUT_COST_PER_TOKEN = 0.00003  # $0.03 per 1K tokens for GPT-4 input
OPENAI_OUTPUT_COST_PER_TOKEN = 0.00006  # $0.06 per 1K tokens for GPT-4 output

# Cache Configuration
CACHE_DURATION = 3600  # 1 hour in seconds
CACHE_DIR = ".cache"

# Prediction Configuration
DEFAULT_ITERATIONS = 1000
MIN_CONFIDENCE_SCORE = 0.7
MAX_RETRIES = 3

# Model Configuration
CLAUDE_MODEL = "claude-3-opus-20240229"
GROK_MODEL = "grok-1"  # Example model name, adjust based on actual API
OPENAI_MODEL = "gpt-4"  # or "gpt-3.5-turbo" for a cheaper option

# API Configuration
CLAUDE_MAX_TOKENS = 1000
CLAUDE_TEMPERATURE = 0.7
GROK_MAX_TOKENS = 1000
GROK_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 1000
OPENAI_TEMPERATURE = 0.7 