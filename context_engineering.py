import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import anthropic
import openai
import requests
import json
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
import os
from config import *

@dataclass
class PredictionContext:
    date: datetime
    historical_numbers: List[List[int]]
    patterns: Dict[str, List[int]]
    market_conditions: Dict[str, float]
    ai_enabled: bool = False
    model_name: str = "none"  # "claude", "grok", "gpt-4", or "none"

@dataclass
class AIResponse:
    prediction: List[int]
    explanation: str
    confidence_score: float
    cost: float

class GrokAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.grok.com/v1"  # Replace with actual Grok API endpoint
        
    def create_completion(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": GROK_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{self.base_url}/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Grok API error: {response.text}")
            
        return response.json()

class OpenAIAPI:
    def __init__(self, api_key: str):
        openai.api_key = api_key
    
    def create_completion(self, prompt: str, model: str = "gpt-4") -> dict:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a lottery number prediction expert. Provide predictions in JSON format with numbers, explanation, and confidence score."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

class ContextManager:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize API clients with error handling
        try:
            self.claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None
        except Exception as e:
            print(f"Warning: Failed to initialize Claude client - {str(e)}")
            self.claude_client = None
            
        try:
            self.grok_client = GrokAPI(api_key=GROK_API_KEY) if GROK_API_KEY else None
        except Exception as e:
            print(f"Warning: Failed to initialize Grok client - {str(e)}")
            self.grok_client = None
            
        try:
            self.openai_client = OpenAIAPI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI client - {str(e)}")
            self.openai_client = None
            
        self.current_context: Optional[PredictionContext] = None
        self.cache_duration = CACHE_DURATION
        
    def _generate_cache_key(self, context: PredictionContext) -> str:
        """Generate a unique cache key based on context parameters"""
        context_str = f"{context.date}_{str(context.historical_numbers)}_{str(context.patterns)}"
        return hashlib.md5(context_str.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[AIResponse]:
        """Retrieve cached AI response if available and not expired"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                timestamp, response = pickle.load(f)
                if (datetime.now() - timestamp).total_seconds() < self.cache_duration:
                    return response
        return None

    def _cache_response(self, cache_key: str, response: AIResponse):
        """Cache AI response with timestamp"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump((datetime.now(), response), f)

    def create_context(self, 
                      historical_data: pd.DataFrame, 
                      target_date: datetime,
                      ai_enabled: bool = False) -> PredictionContext:
        """Create prediction context from historical data"""
        # Get recent draws (last 10 draws)
        recent_numbers = historical_data.head(10)[
            ['number1', 'number2', 'number3', 'number4', 'number5', 'number6', 'extra']
        ].values.tolist()

        # Calculate comprehensive patterns
        patterns = {
            'hot_numbers': self._get_hot_numbers(historical_data),
            'cold_numbers': self._get_cold_numbers(historical_data),
            'frequent_pairs': self._get_frequent_pairs(historical_data),
            'even_odd_ratio': self._calculate_even_odd_ratio(historical_data),
            'high_low_ratio': self._calculate_high_low_ratio(historical_data),
            'sum_trends': self._calculate_sum_trends(historical_data),
            'sequential_patterns': self._find_sequential_patterns(historical_data),
            'gap_analysis': self._analyze_number_gaps(historical_data),
            'sector_analysis': self._analyze_number_sectors(historical_data)
        }

        # Enhanced market conditions
        market_conditions = {
            'volatility': self._calculate_volatility(historical_data),
            'trend_strength': self._calculate_trend_strength(historical_data),
            'cycle_position': self._calculate_cycle_position(historical_data),
            'pattern_strength': self._calculate_pattern_strength(historical_data)
        }

        self.current_context = PredictionContext(
            date=target_date,
            historical_numbers=recent_numbers,
            patterns=patterns,
            market_conditions=market_conditions,
            ai_enabled=ai_enabled
        )
        
        return self.current_context

    def _get_hot_numbers(self, data: pd.DataFrame, lookback: int = 50) -> List[int]:
        """Identify frequently drawn numbers in recent draws"""
        recent_data = data.head(lookback)
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(recent_data[f'number{i}'].tolist())
        return pd.Series(all_numbers).value_counts().head(10).index.tolist()

    def _get_cold_numbers(self, data: pd.DataFrame, lookback: int = 50) -> List[int]:
        """Identify rarely drawn numbers in recent draws"""
        recent_data = data.head(lookback)
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(recent_data[f'number{i}'].tolist())
        return pd.Series(all_numbers).value_counts().tail(10).index.tolist()

    def _get_frequent_pairs(self, data: pd.DataFrame, lookback: int = 50) -> List[Tuple[int, int]]:
        """Identify frequently occurring number pairs"""
        recent_data = data.head(lookback)
        pairs = []
        for _, row in recent_data.iterrows():
            numbers = [row[f'number{i}'] for i in range(1, 7)]
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pairs.append(tuple(sorted([numbers[i], numbers[j]])))
        return pd.Series(pairs).value_counts().head(10).index.tolist()

    def _calculate_volatility(self, data: pd.DataFrame, window: int = 10) -> float:
        """Calculate number volatility based on recent draws"""
        recent_sums = data.head(window)[
            ['number1', 'number2', 'number3', 'number4', 'number5', 'number6']
        ].sum(axis=1)
        return recent_sums.std()

    def _calculate_trend_strength(self, data: pd.DataFrame, window: int = 10) -> float:
        """Calculate trend strength based on recent draws"""
        recent_sums = data.head(window)[
            ['number1', 'number2', 'number3', 'number4', 'number5', 'number6']
        ].sum(axis=1)
        trend = np.polyfit(range(len(recent_sums)), recent_sums, 1)[0]
        return abs(trend)

    def _calculate_even_odd_ratio(self, data: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """Calculate the ratio of even to odd numbers in recent draws"""
        recent_data = data.head(lookback)
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(recent_data[f'number{i}'].tolist())
        
        even_count = sum(1 for num in all_numbers if num % 2 == 0)
        total_count = len(all_numbers)
        
        return {
            'even_ratio': even_count / total_count,
            'odd_ratio': (total_count - even_count) / total_count
        }

    def _calculate_high_low_ratio(self, data: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """Calculate the ratio of high (26-49) to low (1-25) numbers"""
        recent_data = data.head(lookback)
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(recent_data[f'number{i}'].tolist())
        
        high_count = sum(1 for num in all_numbers if num > 25)
        total_count = len(all_numbers)
        
        return {
            'high_ratio': high_count / total_count,
            'low_ratio': (total_count - high_count) / total_count
        }

    def _calculate_sum_trends(self, data: pd.DataFrame, window: int = 10) -> Dict[str, float]:
        """Analyze trends in the sum of numbers"""
        recent_sums = data.head(window)[
            ['number1', 'number2', 'number3', 'number4', 'number5', 'number6']
        ].sum(axis=1)
        
        return {
            'average_sum': recent_sums.mean(),
            'sum_trend': np.polyfit(range(len(recent_sums)), recent_sums, 1)[0],
            'sum_volatility': recent_sums.std()
        }

    def _find_sequential_patterns(self, data: pd.DataFrame, lookback: int = 20) -> Dict[str, List[int]]:
        """Find sequential number patterns in recent draws"""
        recent_data = data.head(lookback)
        sequential_patterns = []
        
        for _, row in recent_data.iterrows():
            numbers = sorted([row[f'number{i}'] for i in range(1, 7)])
            for i in range(len(numbers)-1):
                if numbers[i+1] - numbers[i] == 1:
                    sequential_patterns.append((numbers[i], numbers[i+1]))
        
        pattern_counts = pd.Series(sequential_patterns).value_counts()
        
        return {
            'common_sequences': pattern_counts.head(5).index.tolist(),
            'sequence_frequency': pattern_counts.head(5).tolist()
        }

    def _analyze_number_gaps(self, data: pd.DataFrame, lookback: int = 30) -> Dict[str, List[int]]:
        """Analyze gaps between consecutive draws"""
        recent_data = data.head(lookback)
        gaps = []
        
        for i in range(len(recent_data)-1):
            current_numbers = set([recent_data.iloc[i][f'number{j}'] for j in range(1, 7)])
            next_numbers = set([recent_data.iloc[i+1][f'number{j}'] for j in range(1, 7)])
            gaps.append(len(current_numbers - next_numbers))
        
        return {
            'average_gap': np.mean(gaps),
            'gap_trend': np.polyfit(range(len(gaps)), gaps, 1)[0] if gaps else 0,
            'common_gaps': pd.Series(gaps).value_counts().head(3).index.tolist()
        }

    def _analyze_number_sectors(self, data: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """Analyze number distribution across sectors (1-10, 11-20, etc.)"""
        recent_data = data.head(lookback)
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(recent_data[f'number{i}'].tolist())
        
        sectors = {
            '1-10': sum(1 for n in all_numbers if 1 <= n <= 10),
            '11-20': sum(1 for n in all_numbers if 11 <= n <= 20),
            '21-30': sum(1 for n in all_numbers if 21 <= n <= 30),
            '31-40': sum(1 for n in all_numbers if 31 <= n <= 40),
            '41-49': sum(1 for n in all_numbers if 41 <= n <= 49)
        }
        
        total = len(all_numbers)
        return {k: v/total for k, v in sectors.items()}

    def _calculate_cycle_position(self, data: pd.DataFrame, cycle_length: int = 20) -> float:
        """Calculate position in the theoretical cycle"""
        total_draws = len(data)
        return (total_draws % cycle_length) / cycle_length

    def _calculate_pattern_strength(self, data: pd.DataFrame, lookback: int = 30) -> float:
        """Calculate the strength of identified patterns"""
        recent_data = data.head(lookback)
        pattern_matches = 0
        total_comparisons = 0
        
        for i in range(len(recent_data)-1):
            current_numbers = set([recent_data.iloc[i][f'number{j}'] for j in range(1, 7)])
            next_numbers = set([recent_data.iloc[i+1][f'number{j}'] for j in range(1, 7)])
            common_numbers = len(current_numbers.intersection(next_numbers))
            pattern_matches += common_numbers
            total_comparisons += 6
        
        return pattern_matches / total_comparisons if total_comparisons > 0 else 0

    def get_ai_prediction(self, model: str = "claude") -> AIResponse:
        """Get AI prediction using selected model with enhanced context"""
        if not self.current_context or not self.current_context.ai_enabled:
            raise ValueError("Context not initialized or AI not enabled")

        # Check if selected model's client is available
        if model == "claude" and not self.claude_client:
            raise ValueError("Claude client not initialized. Please check your API key.")
        elif model == "gpt-4" and not self.openai_client:
            raise ValueError("OpenAI client not initialized. Please check your API key.")
        elif model == "grok" and not self.grok_client:
            raise ValueError("Grok client not initialized. Please check your API key.")

        self.current_context.model_name = model
        cache_key = self._generate_cache_key(self.current_context)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        # Prepare enhanced context for AI
        context_message = f"""
        Based on the following detailed lottery analysis, predict the next 6 numbers and 1 extra number:

        1. Recent Draw History:
        Last 5 draws: {self.current_context.historical_numbers[:5]}
        
        2. Number Frequency Analysis:
        - Hot numbers (most frequent): {self.current_context.patterns['hot_numbers']}
        - Cold numbers (least frequent): {self.current_context.patterns['cold_numbers']}
        - Common pairs: {self.current_context.patterns['frequent_pairs']}
        
        3. Distribution Patterns:
        - Even/Odd ratio: {self.current_context.patterns['even_odd_ratio']}
        - High/Low ratio: {self.current_context.patterns['high_low_ratio']}
        - Sector distribution: {self.current_context.patterns['sector_analysis']}
        
        4. Sequential Analysis:
        - Common sequences: {self.current_context.patterns['sequential_patterns']['common_sequences']}
        - Sequence frequency: {self.current_context.patterns['sequential_patterns']['sequence_frequency']}
        
        5. Gap Analysis:
        - Average gap: {self.current_context.patterns['gap_analysis']['average_gap']:.2f}
        - Gap trend: {self.current_context.patterns['gap_analysis']['gap_trend']:.2f}
        - Common gaps: {self.current_context.patterns['gap_analysis']['common_gaps']}
        
        6. Sum Trends:
        - Average sum: {self.current_context.patterns['sum_trends']['average_sum']:.2f}
        - Sum trend: {self.current_context.patterns['sum_trends']['sum_trend']:.2f}
        - Sum volatility: {self.current_context.patterns['sum_trends']['sum_volatility']:.2f}
        
        7. Market Conditions:
        - Volatility: {self.current_context.market_conditions['volatility']:.2f}
        - Trend strength: {self.current_context.market_conditions['trend_strength']:.2f}
        - Cycle position: {self.current_context.market_conditions['cycle_position']:.2f}
        - Pattern strength: {self.current_context.market_conditions['pattern_strength']:.2f}
        
        Target date: {self.current_context.date}

        Requirements:
        - Predict 6 main numbers (1-49) and 1 extra number
        - Main numbers must be unique
        - Extra number can match a main number
        - Provide detailed reasoning based on the patterns above
        - Include confidence score (0-1) based on pattern strength
        
        Provide your response in the following JSON format:
        {{
            "prediction": [num1, num2, num3, num4, num5, num6, extra],
            "explanation": "Your detailed reasoning here",
            "confidence": 0.XX
        }}
        """

        try:
            if model == "claude":
                if not self.claude_client:
                    raise ValueError("Claude client not initialized")
                response = self.claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    temperature=CLAUDE_TEMPERATURE,
                    messages=[
                        {"role": "user", "content": context_message}
                    ]
                )
                response_text = response.content[0].text
                cost = len(context_message.split()) * CLAUDE_COST_PER_TOKEN

            elif model == "gpt-4":
                if not self.openai_client:
                    raise ValueError("OpenAI client not initialized")
                response = self.openai_client.create_completion(
                    prompt=context_message,
                    model=OPENAI_MODEL
                )
                response_text = response.choices[0].message.content
                # Calculate cost based on input and output tokens
                input_tokens = len(context_message.split()) * 0.75  # Approximate token count
                output_tokens = len(response_text.split()) * 0.75
                cost = (input_tokens * OPENAI_INPUT_COST_PER_TOKEN + 
                       output_tokens * OPENAI_OUTPUT_COST_PER_TOKEN)

            else:  # model == "grok"
                if not self.grok_client:
                    raise ValueError("Grok client not initialized")
                response = self.grok_client.create_completion(
                    prompt=context_message,
                    max_tokens=GROK_MAX_TOKENS,
                    temperature=GROK_TEMPERATURE
                )
                response_text = response["choices"][0]["text"]
                cost = len(context_message.split()) * GROK_COST_PER_TOKEN

            # Parse the JSON response
            try:
                parsed_response = json.loads(response_text)
                prediction = parsed_response["prediction"]
                explanation = parsed_response["explanation"]
                confidence_score = parsed_response["confidence"]
                
                # Validate prediction
                if len(prediction) != 7:
                    raise ValueError("Prediction must contain exactly 7 numbers")
                
                main_numbers = prediction[:6]
                if len(set(main_numbers)) != 6:
                    raise ValueError("Main numbers must be unique")
                
                if not all(1 <= n <= 49 for n in prediction):
                    raise ValueError("All numbers must be between 1 and 49")
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Fallback parsing if JSON is malformed
                # st.warning(f"Warning: AI response parsing error - {str(e)}") # This line was commented out in the new_code, so I'm removing it.
                prediction = [1, 2, 3, 4, 5, 6, 7]  # Default prediction
                explanation = response_text
                confidence_score = 0.5

            ai_response = AIResponse(
                prediction=prediction,
                explanation=explanation,
                confidence_score=confidence_score,
                cost=cost
            )

            self._cache_response(cache_key, ai_response)
            return ai_response

        except Exception as e:
            raise Exception(f"AI prediction failed: {str(e)}")

class BulkPredictor:
    def __init__(self, analyzer, context_manager: ContextManager):
        self.analyzer = analyzer
        self.context_manager = context_manager
        
    def predict_bulk(self, 
                    historical_data: pd.DataFrame,
                    target_date: datetime,
                    iterations: int = 1000,
                    ai_enabled: bool = False,
                    model: str = "claude") -> Dict:
        """Perform bulk predictions and analyze results"""
        all_predictions = []
        total_cost = 0
        
        for _ in range(iterations):
            if ai_enabled:
                context = self.context_manager.create_context(
                    historical_data=historical_data,
                    target_date=target_date,
                    ai_enabled=True
                )
                response = self.context_manager.get_ai_prediction(model=model)
                prediction = response.prediction
                total_cost += response.cost
            else:
                last_numbers = historical_data.iloc[0][
                    ['number1', 'number2', 'number3', 'number4', 'number5', 'number6', 'extra']
                ].values
                prediction, _ = self.analyzer.predict_next_numbers(last_numbers, target_date)
            
            all_predictions.append(prediction)
        
        # Analyze predictions
        flat_predictions = [num for pred in all_predictions for num in pred]
        number_counts = pd.Series(flat_predictions).value_counts()
        
        return {
            'top_8_numbers': number_counts.head(8).index.tolist(),
            'next_12_numbers': number_counts.iloc[8:20].index.tolist(),
            'number_frequencies': number_counts.to_dict(),
            'total_cost': total_cost if ai_enabled else 0,
            'average_cost_per_prediction': total_cost / iterations if ai_enabled else 0
        } 