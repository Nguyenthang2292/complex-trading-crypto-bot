import logging
import numpy as np
import os
import sys
import pandas as pd
from typing import List, Dict

# Add the parent directory to sys.path to allow importing modules from sibling directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from livetrade.config import (DEFAULT_WINDOW_SIZE, RSI_PERIOD)
from utilities._logger import setup_logging

# Setup logging with new format
logger = setup_logging(module_name="BPSs__class__PerformanceAnalyzer", log_level=logging.DEBUG)

class PerformanceAnalyzer:
    """Handles crypto performance analysis for both LONG and SHORT signals."""
    
    def __init__(self):
        self.long_weights = {
            'return': 0.25, 'sharpe': 0.20, 'momentum_short': 0.15, 'momentum_long': 0.15,
            'volume_trend': 0.10, 'low_volatility': 0.10, 'drawdown': 0.05
        }
        self.short_weights = {
            'negative_return': 0.30, 'bearish_momentum_short': 0.20, 'bearish_momentum_long': 0.15,
            'high_rsi': 0.10, 'volume_spike': 0.10, 'distance_from_high': 0.10, 'high_volatility': 0.05
        }
        self.timeframe_weights = {'1h': 0.2, '4h': 0.5, '1d': 0.3}
    
    def calculate_basic_metrics(self, df: pd.DataFrame, period: int) -> Dict:
        """Calculate basic price and volume metrics from historical data."""
        recent_data = df.tail(period).copy()
        
        start_price = recent_data['close'].iloc[0]
        end_price = recent_data['close'].iloc[-1]
        high_price = recent_data['high'].max()
        low_price = recent_data['low'].min()
        
        total_return = (end_price - start_price) / start_price
        daily_returns = recent_data['close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(len(daily_returns)) if len(daily_returns) > 1 else 0
        
        avg_volume = recent_data['volume'].mean()
        avg_price = recent_data['close'].mean()
        avg_volume_usdt = avg_volume * avg_price
        
        momentum_short = self._calculate_momentum(recent_data['close'], 5, 10)
        momentum_long = self._calculate_momentum(recent_data['close'], 10, DEFAULT_WINDOW_SIZE)
        max_drawdown = self._calculate_max_drawdown(recent_data['close'])
        sharpe_ratio = total_return / volatility if volatility > 0 else 0
        rsi = self._calculate_simple_rsi(recent_data['close'], period=RSI_PERIOD)
        volume_trend = self._calculate_momentum(recent_data['volume'], 5, 10)
        
        recent_high = recent_data['high'].rolling(window=min(DEFAULT_WINDOW_SIZE//2, len(recent_data))).max().iloc[-1]
        distance_from_high = (recent_high - end_price) / recent_high
        
        return {
            'start_price': start_price, 'end_price': end_price, 'high_price': high_price, 'low_price': low_price,
            'total_return': total_return, 'volatility': volatility, 'avg_volume_usdt': avg_volume_usdt,
            'momentum_short': momentum_short, 'momentum_long': momentum_long, 'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio, 'rsi': rsi, 'volume_trend': volume_trend, 'distance_from_high': distance_from_high
        }
    
    def calculate_long_score(self, metrics: Dict) -> float:
        """Calculate composite score for LONG signals based on weighted metrics."""
        normalized_return = max(0, min(1, (metrics['total_return'] + 0.5) / 1.0))
        normalized_sharpe = max(0, min(1, (metrics['sharpe_ratio'] + 2) / 4))
        normalized_momentum_short = max(0, min(1, (metrics['momentum_short'] + 0.2) / 0.4))
        normalized_momentum_long = max(0, min(1, (metrics['momentum_long'] + 0.2) / 0.4))
        normalized_volume_trend = max(0, min(1, (metrics['volume_trend'] + 0.5) / 1.0))
        normalized_low_volatility = max(0, min(1, 1 - (metrics['volatility'] / 2)))
        normalized_drawdown = max(0, min(1, 1 - abs(metrics['max_drawdown'])))
        
        return (
            self.long_weights['return'] * normalized_return +
            self.long_weights['sharpe'] * normalized_sharpe +
            self.long_weights['momentum_short'] * normalized_momentum_short +
            self.long_weights['momentum_long'] * normalized_momentum_long +
            self.long_weights['volume_trend'] * normalized_volume_trend +
            self.long_weights['low_volatility'] * normalized_low_volatility +
            self.long_weights['drawdown'] * normalized_drawdown
        )
    
    def calculate_short_score(self, metrics: Dict) -> float:
        """Calculate composite score for SHORT signals based on weighted metrics."""
        normalized_negative_return = max(0, min(1, (-metrics['total_return'] + 0.1) / 0.6))
        normalized_bearish_momentum_short = max(0, min(1, (-metrics['momentum_short'] + 0.1) / 0.3))
        normalized_bearish_momentum_long = max(0, min(1, (-metrics['momentum_long'] + 0.1) / 0.3))
        normalized_high_rsi = max(0, min(1, (metrics['rsi'] - 50) / 50))
        normalized_volume_spike = max(0, min(1, (metrics['volume_trend'] + 0.5) / 1.0))
        normalized_distance_from_high = max(0, min(1, metrics['distance_from_high'] / 0.3))
        high_volatility_score = min(1.0, metrics['volatility'] * 2)
        
        return (
            self.short_weights['negative_return'] * normalized_negative_return +
            self.short_weights['bearish_momentum_short'] * normalized_bearish_momentum_short +
            self.short_weights['bearish_momentum_long'] * normalized_bearish_momentum_long +
            self.short_weights['high_rsi'] * normalized_high_rsi +
            self.short_weights['volume_spike'] * normalized_volume_spike +
            self.short_weights['distance_from_high'] * normalized_distance_from_high +
            self.short_weights['high_volatility'] * high_volatility_score
        )
    
    def calculate_performance_metrics(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int) -> Dict:
        """Calculate comprehensive performance metrics for both LONG and SHORT signals."""
        try:
            if len(df) < period:
                raise ValueError(f"Insufficient data: {len(df)} < {period}")
            
            basic_metrics = self.calculate_basic_metrics(df, period)
            long_score = self.calculate_long_score(basic_metrics)
            short_score = self.calculate_short_score(basic_metrics)
            
            return {
                'symbol': symbol, 'timeframe': timeframe, 'composite_score': long_score,
                'short_composite_score': short_score, 'total_return': basic_metrics['total_return'],
                'volatility': basic_metrics['volatility'], 'sharpe_ratio': basic_metrics['sharpe_ratio'],
                'max_drawdown': basic_metrics['max_drawdown'], 'momentum_short': basic_metrics['momentum_short'],
                'momentum_long': basic_metrics['momentum_long'], 'bearish_momentum_short': -basic_metrics['momentum_short'],
                'bearish_momentum_long': -basic_metrics['momentum_long'], 'volume_trend': basic_metrics['volume_trend'],
                'avg_volume_usdt': basic_metrics['avg_volume_usdt'], 'rsi': basic_metrics['rsi'],
                'distance_from_high': basic_metrics['distance_from_high'],
                'price_range': {
                    'start': basic_metrics['start_price'], 'end': basic_metrics['end_price'],
                    'high': basic_metrics['high_price'], 'low': basic_metrics['low_price']
                }
            }
        
        except Exception as e:
            logger.debug(f"Error calculating metrics for {symbol}: {e}")
            return {
                'symbol': symbol, 'timeframe': timeframe, 'composite_score': 0,
                'short_composite_score': 0, 'total_return': 0, 'volatility': 999, 'avg_volume_usdt': 0
            }
    
    def calculate_overall_scores(self, symbol_scores: Dict, timeframes: List[str]) -> List[Dict]:
        """Calculate overall composite scores across all timeframes."""
        try:
            overall_results = []
            
            for symbol, tf_scores in symbol_scores.items():
                if len(tf_scores) != len(timeframes):
                    continue
                
                weighted_score = total_weight = 0
                
                for tf in timeframes:
                    if tf in tf_scores and tf in self.timeframe_weights:
                        weight = self.timeframe_weights[tf]
                        weighted_score += tf_scores[tf] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    overall_results.append({
                        'symbol': symbol,
                        'composite_score': weighted_score / total_weight,
                        'timeframe_scores': tf_scores,
                        'score_consistency': np.std(list(tf_scores.values()))
                    })
            
            overall_results.sort(key=lambda x: x['composite_score'], reverse=True)
            return overall_results
        
        except Exception as e:
            logger.error(f"Error calculating overall scores: {e}")
            return []
    
    @staticmethod
    def _calculate_momentum(series: pd.Series, short_period: int, long_period: int) -> float:
        """Calculate momentum between short and long periods."""
        try:
            if len(series) < long_period:
                return 0
            short_avg = series.iloc[-short_period:].mean()
            long_avg = series.iloc[-long_period:-short_period].mean()
            return (short_avg - long_avg) / long_avg if long_avg != 0 else 0
        except:
            return 0
    
    @staticmethod
    def _calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown from a price series."""
        try:
            if len(prices) <= 1:
                return 0.0
            
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            result = drawdown.min()
            
            return result if not pd.isna(result) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def _calculate_simple_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> float:
        """Calculate Relative Strength Index (RSI) for a price series."""
        try:
            if len(prices) < period + 1:
                return 50
            
            delta = prices.diff().astype(float)
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50