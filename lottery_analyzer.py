import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
import random

class LotteryAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = []
        self.features = ['dayofweek', 'month', 'day', 'year',
                        'last_num1', 'last_num2', 'last_num3', 'last_num4', 'last_num5', 'last_num6',
                        'avg_num1', 'avg_num2', 'avg_num3', 'avg_num4', 'avg_num5', 'avg_num6',
                        'last_extra', 'avg_extra']
        self.prediction_history = []
    
    def prepare_features(self, df):
        """Prepare features for prediction"""
        df = df.copy()
        
        # Convert date to datetime if it's not already
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract date features
        df['dayofweek'] = df['date'].dt.weekday
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['year'] = df['date'].dt.year
        
        # Calculate rolling averages and last numbers for main numbers
        for i in range(1, 7):
            col = f'number{i}'
            df[f'avg_num{i}'] = df[col].rolling(window=5, min_periods=1).mean()
            df[f'last_num{i}'] = df[col].shift(1)
        
        # Calculate rolling averages and last numbers for extra number
        df['avg_extra'] = df['extra'].rolling(window=5, min_periods=1).mean()
        df['last_extra'] = df['extra'].shift(1)
        
        # Fill NaN values with mean
        df = df.fillna(df.mean())
        
        return df

    def detect_sequential_patterns(self, data):
        """Detect sequential patterns in the data"""
        patterns = {
            'plus_one_sequences': [],
            'arithmetic_progressions': [],
            'repeating_increments': [],
            'position_correlations': {}
        }
        
        if data.empty:
            return patterns
        
        # Analyze sequential patterns between consecutive draws
        for i in range(len(data) - 1):
            current_draw = [data.iloc[i][f'number{j}'] for j in range(1, 6)]
            next_draw = [data.iloc[i+1][f'number{j}'] for j in range(1, 6)]
            
            # Check for +1 patterns
            plus_one_count = 0
            for curr_num in current_draw:
                if (curr_num + 1) in next_draw:
                    plus_one_count += 1
            
            if plus_one_count > 0:
                patterns['plus_one_sequences'].append({
                    'date': data.iloc[i]['date'],
                    'current': current_draw,
                    'next': next_draw,
                    'plus_one_count': plus_one_count
                })
            
            # Analyze position-based correlations
            for pos in range(5):
                curr_val = current_draw[pos]
                next_val = next_draw[pos]
                diff = next_val - curr_val
                
                if f'position_{pos+1}' not in patterns['position_correlations']:
                    patterns['position_correlations'][f'position_{pos+1}'] = {
                        'mean_diff': diff,
                        'common_diffs': {diff: 1}
                    }
                else:
                    pos_stats = patterns['position_correlations'][f'position_{pos+1}']
                    pos_stats['mean_diff'] = (pos_stats['mean_diff'] + diff) / 2
                    pos_stats['common_diffs'][diff] = pos_stats['common_diffs'].get(diff, 0) + 1
        
        return patterns

    def predict_with_sequential_logic(self, last_numbers, target_date, use_patterns=True):
        """Enhanced prediction using sequential pattern logic with weekday consideration"""
        base_predictions, base_extra = self.predict_next_numbers(last_numbers, target_date)
        
        if not use_patterns:
            return base_predictions, base_extra
        
        # Get weekday-specific patterns if available
        weekday = target_date.weekday()
        weekday_name = target_date.strftime('%A')
        
        # Apply sequential logic based on patterns for main numbers
        enhanced_prediction = []
        used_numbers = set()
        
        for base_num in base_predictions:
            # Add some randomness based on observed patterns
            potential_numbers = [
                base_num,
                min(49, base_num + 1),  # +1 pattern
                max(1, base_num - 1),   # -1 pattern
                min(49, base_num + 2),  # +2 pattern
            ]
            
            # Add weekday-specific numbers if available
            if hasattr(self, 'patterns') and weekday_name in self.patterns.get('weekday_patterns', {}):
                weekday_patterns = self.patterns['weekday_patterns'][weekday_name]
                for avg_num in weekday_patterns.get('avg_numbers', []):
                    avg_num = round(avg_num)
                    if 1 <= avg_num <= 49:
                        potential_numbers.append(avg_num)
            
            # Filter valid numbers (1-49) and avoid duplicates
            valid_numbers = [n for n in potential_numbers if 1 <= n <= 49 and n not in used_numbers]
            
            if valid_numbers:
                # Create weights array of the same size as valid_numbers
                num_weights = len(valid_numbers)
                weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1][:num_weights])
                # If we have more numbers than weights, pad with minimum weight
                if len(weights) < num_weights:
                    weights = np.pad(weights, (0, num_weights - len(weights)), constant_values=0.1)
                # Normalize weights
                weights = weights / np.sum(weights)
                # Choose number
                chosen = np.random.choice(valid_numbers, p=weights)
                enhanced_prediction.append(chosen)
                used_numbers.add(chosen)
            else:
                # Fallback to base prediction
                enhanced_prediction.append(base_num)
                used_numbers.add(base_num)
        
        # Handle extra number similarly but allow it to be same as main numbers
        potential_extra = [
            base_extra,
            min(49, base_extra + 1),
            max(1, base_extra - 1),
            min(49, base_extra + 2)
        ]
        
        # Add weekday-specific numbers for extra
        if hasattr(self, 'patterns') and weekday_name in self.patterns.get('weekday_patterns', {}):
            weekday_patterns = self.patterns['weekday_patterns'][weekday_name]
            if 'avg_extra' in weekday_patterns:
                avg_extra = round(weekday_patterns['avg_extra'])
                if 1 <= avg_extra <= 49:
                    potential_extra.append(avg_extra)
        
        # Filter valid extra numbers (1-49)
        valid_extra = [n for n in potential_extra if 1 <= n <= 49]
        
        if valid_extra:
            # Create weights array of the same size as valid_extra
            num_weights = len(valid_extra)
            weights = np.array([0.4, 0.3, 0.2, 0.1][:num_weights])
            # If we have more numbers than weights, pad with minimum weight
            if len(weights) < num_weights:
                weights = np.pad(weights, (0, num_weights - len(weights)), constant_values=0.1)
            # Normalize weights
            weights = weights / np.sum(weights)
            # Choose extra number
            enhanced_extra = np.random.choice(valid_extra, p=weights)
        else:
            enhanced_extra = base_extra
        
        return sorted(enhanced_prediction), enhanced_extra

    def predict_multiple_sets(self, last_numbers, target_date, count=50, use_patterns=True):
        """Generate multiple prediction sets with weekday consideration"""
        predictions = []
        weekday_name = target_date.strftime('%A')
        
        for i in range(count):
            if use_patterns:
                pred, extra_pred = self.predict_with_sequential_logic(last_numbers, target_date, use_patterns=True)
            else:
                pred, extra_pred = self.predict_next_numbers(last_numbers, target_date)
            
            # Add some variation for multiple predictions
            if i > 0:
                # Introduce slight randomness for variety
                for j in range(len(pred)):
                    if random.random() < 0.3:  # 30% chance to modify
                        adjustment = random.choice([-2, -1, 1, 2])
                        new_val = pred[j] + adjustment
                        if 1 <= new_val <= 39 and new_val not in pred:
                            pred[j] = new_val
            
            predictions.append({
                'set_number': i + 1,
                'numbers': sorted(pred),
                'prediction_method': 'pattern_based' if use_patterns else 'ml_based',
                'weekday': weekday_name
            })
        
        return predictions

    def reroll_prediction(self, last_numbers, target_date, previous_prediction=None):
        """Generate a new prediction (reroll)"""
        # Use different random seed for variety
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        new_prediction, new_extra = self.predict_with_sequential_logic(last_numbers, target_date)
        
        # Ensure it's different from previous prediction if provided
        if previous_prediction and new_prediction == previous_prediction:
            # Force some variation
            for i in range(len(new_prediction)):
                if random.random() < 0.4:
                    adjustment = random.choice([-1, 1, 2])
                    new_val = new_prediction[i] + adjustment
                    if 1 <= new_val <= 39 and new_val not in new_prediction:
                        new_prediction[i] = new_val
        
        # Store in history
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': new_prediction,
            'extra': new_extra,
            'method': 'reroll'
        })
        
        return sorted(new_prediction), new_extra

    def analyze_prediction_confidence(self, predictions_list):
        """Analyze confidence in predictions based on frequency"""
        number_frequency = {}
        
        for pred_set in predictions_list:
            for num in pred_set['numbers']:
                number_frequency[num] = number_frequency.get(num, 0) + 1
        
        # Calculate confidence scores
        total_sets = len(predictions_list)
        confidence_scores = {
            num: (freq / total_sets) * 100 
            for num, freq in number_frequency.items()
        }
        
        return {
            'most_confident': sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:10],
            'frequency_distribution': number_frequency,
            'confidence_scores': confidence_scores
        }

    def train_models(self, data):
        """Train separate models for each number with weekday consideration"""
        df = self.prepare_features(data)
        
        # Store patterns for later use
        self.patterns = self.analyze_patterns(df)
        
        X = df[self.features]
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.models = []
        # Train models for main numbers
        for i in range(1, 7):
            y = df[f'number{i}']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            self.models.append(model)
        
        # Train model for extra number
        y_extra = df['extra']
        model_extra = RandomForestRegressor(n_estimators=100, random_state=42)
        model_extra.fit(X_scaled, y_extra)
        self.models.append(model_extra)
        
        return self.models

    def predict_next_numbers(self, last_numbers, target_date):
        """Predict numbers for the next draw"""
        # Prepare features for prediction
        pred_features = pd.DataFrame({
            'dayofweek': [target_date.weekday()],
            'month': [target_date.month],
            'day': [target_date.day],
            'year': [target_date.year]
        })
        
        # Add last numbers and their averages for main numbers
        for i in range(6):
            pred_features[f'last_num{i+1}'] = last_numbers[i]
            pred_features[f'avg_num{i+1}'] = last_numbers[i]  # Using last number as average
        
        # Add last extra number and its average
        pred_features['last_extra'] = last_numbers[6]  # Last position is extra number
        pred_features['avg_extra'] = last_numbers[6]  # Using last number as average
        
        # Scale features
        X_pred = self.scaler.transform(pred_features[self.features])
        
        # Make predictions for main numbers
        predictions = []
        used_numbers = set()
        
        # Predict main numbers (1-49 range for Mark 6)
        for model in self.models[:-1]:  # All but last model (which is for extra number)
            pred = model.predict(X_pred)[0]
            # Round to nearest valid number (1-49) and ensure no duplicates
            while True:
                num = max(1, min(49, round(pred)))
                if num not in used_numbers:
                    predictions.append(num)
                    used_numbers.add(num)
                    break
                pred += 1
        
        # Predict extra number (1-49 range, can be same as main numbers)
        extra_pred = max(1, min(49, round(self.models[-1].predict(X_pred)[0])))
        
        # Return sorted main numbers and extra number
        return sorted(predictions), extra_pred

    def analyze_patterns(self, df):
        """Analyze historical patterns with weekday consideration"""
        patterns = {
            'weekday_patterns': {},
            'general_patterns': self.detect_sequential_patterns(df),
            'day_statistics': {
                'number1': {},
                'number2': {},
                'number3': {},
                'number4': {},
                'number5': {},
                'number6': {}
            },
            'sequential_patterns': {
                'plus_one_sequences': [],
                'arithmetic_progressions': [],
                'repeating_increments': [],
                'position_correlations': {}
            }
        }
        
        # Analyze patterns for each weekday
        for day in range(7):  # 0-6 (Monday to Sunday)
            day_data = df[df['date'].dt.weekday == day]
            if not day_data.empty:
                weekday_name = day_data.iloc[0]['date'].strftime('%A')
                day_sequential_patterns = self.detect_sequential_patterns(day_data)
                
                patterns['weekday_patterns'][weekday_name] = {
                    'sequential_patterns': day_sequential_patterns,
                    'avg_numbers': [
                        day_data[f'number{i}'].mean() for i in range(1, 7)
                    ],
                    'most_common': [
                        day_data[f'number{i}'].value_counts().head(3).to_dict() 
                        for i in range(1, 7)
                    ]
                }
                
                # Add day statistics in the correct format
                for num in range(1, 7):
                    patterns['day_statistics'][f'number{num}'][day] = day_data[f'number{num}'].mean()
                
                # Update sequential patterns
                patterns['sequential_patterns']['plus_one_sequences'].extend(day_sequential_patterns['plus_one_sequences'])
                patterns['sequential_patterns']['arithmetic_progressions'].extend(day_sequential_patterns.get('arithmetic_progressions', []))
                patterns['sequential_patterns']['repeating_increments'].extend(day_sequential_patterns.get('repeating_increments', []))
                
                # Merge position correlations
                for pos, corr in day_sequential_patterns.get('position_correlations', {}).items():
                    if pos not in patterns['sequential_patterns']['position_correlations']:
                        patterns['sequential_patterns']['position_correlations'][pos] = corr
                    else:
                        # Average the correlations
                        existing = patterns['sequential_patterns']['position_correlations'][pos]
                        patterns['sequential_patterns']['position_correlations'][pos] = {
                            'mean_diff': (existing['mean_diff'] + corr['mean_diff']) / 2,
                            'common_diffs': {**existing['common_diffs'], **corr['common_diffs']}
                        }
        
        return patterns

    def save_models(self, filename='lottery_models.joblib'):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'prediction_history': self.prediction_history
        }
        joblib.dump(model_data, filename)
    
    def load_models(self, filename='lottery_models.joblib'):
        """Load trained models"""
        model_data = joblib.load(filename)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.prediction_history = model_data.get('prediction_history', [])

if __name__ == "__main__":
    # Example usage
    analyzer = LotteryAnalyzer()
    
    # Load historical data
    df = pd.read_csv('lottery_history.csv')
    
    # Train models
    analyzer.train_models(df)
    
    # Get last numbers from the most recent draw
    last_numbers = df.iloc[0][['number1', 'number2', 'number3', 'number4', 'number5', 'number6']].values
    
    # Predict next draw
    next_date = datetime.now()
    while next_date.weekday() != 0:  # Find next Monday
        next_date += timedelta(days=1)
    
    predictions, extra_pred = analyzer.predict_next_numbers(last_numbers, next_date)
    print(f"Predicted numbers for {next_date.date()}: {predictions}, Extra: {extra_pred}")
    
    # Analyze patterns
    patterns = analyzer.analyze_patterns(df)
    print("\nPattern Analysis:")
    print(f"Hot numbers: {patterns['hot_numbers']}")
    print(f"Cold numbers: {patterns['cold_numbers']}") 