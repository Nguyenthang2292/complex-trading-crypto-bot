"""
Token Metrics API Main Interface
Interactive command line interface for Token Metrics API
"""

import sys
import os
import json
from typing import Dict

# Add parent directory to path to import utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the same directory
from token_metrics import (
    TokenMetricsAPI, 
    TokenMetricsAPIError,
    create_token_metrics_client,
    get_comprehensive_token_analysis,
    get_multiple_tokens_analysis,
    test_api_connection,
    test_specific_token
)
from utilities.logger import setup_logging

logger = setup_logging(module_name="token_metrics_main", log_level=20)

class TokenMetricsCLI:
    """Interactive CLI for Token Metrics API"""
    
    def __init__(self):
        self.client = None
        self.available_modes = {
            '1': ('Price & Basic Info', self.show_price_info),
            '2': ('Trader Grade', self.show_trader_grade),
            '3': ('Trading Signals', self.show_trading_signals),
            '4': ('Market Sentiment', self.show_market_sentiment),
            '5': ('AI Analysis', self.show_ai_analysis),
            '6': ('Quantitative Metrics', self.show_quantitative_metrics),
            '7': ('Resistance & Support', self.show_resistance_support),
            '8': ('Scenario Analysis', self.show_scenario_analysis),
            '9': ('Comprehensive Analysis', self.show_comprehensive_analysis),
            '10': ('Multiple Tokens Analysis', self.show_multiple_tokens_analysis),
            '11': ('Top Tokens', self.show_top_tokens),
            '12': ('Search Tokens', self.search_tokens),
            '13': ('Performance Indices', self.show_performance_indices),
            '0': ('Exit', self.exit_program)
        }
    
    def initialize_client(self) -> bool:
        """Initialize the Token Metrics API client"""
        try:
            logger.info("Initializing Token Metrics API client...")
            self.client = create_token_metrics_client()
            logger.success("Client initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize client: {str(e)}")
            return False
    
    def get_user_input(self, prompt: str, default: str = "") -> str:
        """Get user input with optional default value"""
        if default:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            return user_input if user_input else default
        else:
            return input(f"{prompt}: ").strip()
    
    def get_crypto_symbol(self) -> str:
        """Get crypto symbol from user input"""
        while True:
            symbol = self.get_user_input("Enter crypto symbol (e.g., BTC, ETH, ADA)").upper()
            if symbol:
                return symbol
            logger.warning("Please enter a valid crypto symbol.")
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🔍 TOKEN METRICS API - CRYPTO ANALYSIS TOOL")
        print("="*60)
        print("Available Analysis Modes:")
        print("-"*40)
        
        for key, (name, _) in self.available_modes.items():
            print(f"{key:>2}. {name}")
        
        print("-"*40)
        print("💡 Tip: Enter '0' to exit the program")
        print("="*60)
    
    def get_user_choice(self) -> str:
        """Get user's menu choice"""
        while True:
            choice = self.get_user_input("Select analysis mode (0-13)")
            if choice in self.available_modes:
                return choice
            logger.warning("Invalid choice. Please select a number between 0-13.")
    
    def format_price_data(self, data: Dict) -> str:
        """Format price data for display"""
        if not data:
            return "No price data available"
        
        result = []
        result.append(f"💰 Price Information for {data.get('symbol', 'Unknown')}")
        result.append("-" * 50)
        
        # Basic price info
        if 'price' in data:
            result.append(f"Current Price: ${data['price']:,.2f}" if isinstance(data['price'], (int, float)) else f"Current Price: {data['price']}")
        
        if 'market_cap' in data:
            result.append(f"Market Cap: ${data['market_cap']:,.0f}" if isinstance(data['market_cap'], (int, float)) else f"Market Cap: {data['market_cap']}")
        
        if 'volume_24h' in data:
            result.append(f"24h Volume: ${data['volume_24h']:,.0f}" if isinstance(data['volume_24h'], (int, float)) else f"24h Volume: {data['volume_24h']}")
        
        if 'change_24h' in data:
            change = data['change_24h']
            if isinstance(change, (int, float)):
                color = "🟢" if change >= 0 else "🔴"
                result.append(f"24h Change: {color} {change:+.2f}%")
            else:
                result.append(f"24h Change: {change}")
        
        return "\n".join(result)
    
    def format_trader_grade(self, data: Dict) -> str:
        """Format trader grade data for display"""
        if not data:
            return "No trader grade data available"
        
        result = []
        result.append(f"📊 Trader Grade Analysis")
        result.append("-" * 50)
        
        if 'grade' in data:
            grade = data['grade']
            result.append(f"Overall Grade: {grade}")
            
            # Add grade interpretation
            grade_interpretation = {
                'A+': "Excellent - Strong buy signal",
                'A': "Very Good - Buy signal",
                'B+': "Good - Moderate buy signal",
                'B': "Fair - Hold signal",
                'C+': "Below Average - Consider selling",
                'C': "Poor - Sell signal",
                'D': "Very Poor - Strong sell signal",
                'F': "Failing - Avoid"
            }
            
            if grade in grade_interpretation:
                result.append(f"Interpretation: {grade_interpretation[grade]}")
        
        if 'score' in data:
            result.append(f"Score: {data['score']}")
        
        if 'factors' in data:
            result.append("\nKey Factors:")
            for factor in data['factors'][:5]:  # Show top 5 factors
                result.append(f"  • {factor}")
        
        return "\n".join(result)
    
    def format_trading_signals(self, data: Dict) -> str:
        """Format trading signals data for display"""
        if not data:
            return "No trading signals available"
        
        result = []
        result.append(f"📈 Trading Signals")
        result.append("-" * 50)
        
        if 'signal' in data:
            signal = data['signal']
            result.append(f"Current Signal: {signal}")
            
            # Add signal interpretation
            signal_interpretation = {
                'BUY': "🟢 Strong Buy Signal",
                'SELL': "🔴 Strong Sell Signal",
                'HOLD': "🟡 Hold Position",
                'STRONG_BUY': "🟢🟢 Very Strong Buy",
                'STRONG_SELL': "🔴🔴 Very Strong Sell"
            }
            
            if signal in signal_interpretation:
                result.append(f"Action: {signal_interpretation[signal]}")
        
        if 'confidence' in data:
            confidence = data['confidence']
            if isinstance(confidence, (int, float)):
                result.append(f"Confidence: {confidence:.1f}%")
            else:
                result.append(f"Confidence: {confidence}")
        
        if 'indicators' in data:
            result.append("\nTechnical Indicators:")
            for indicator, value in data['indicators'].items():
                result.append(f"  • {indicator}: {value}")
        
        return "\n".join(result)
    
    def format_sentiment(self, data: Dict) -> str:
        """Format sentiment data for display"""
        if not data:
            return "No sentiment data available"
        
        result = []
        result.append(f"😊 Market Sentiment Analysis")
        result.append("-" * 50)
        
        if 'sentiment_score' in data:
            score = data['sentiment_score']
            if isinstance(score, (int, float)):
                if score > 0.6:
                    sentiment = "🟢 Bullish"
                elif score < 0.4:
                    sentiment = "🔴 Bearish"
                else:
                    sentiment = "🟡 Neutral"
                result.append(f"Sentiment Score: {score:.2f} ({sentiment})")
            else:
                result.append(f"Sentiment Score: {score}")
        
        if 'social_sentiment' in data:
            result.append(f"Social Sentiment: {data['social_sentiment']}")
        
        if 'news_sentiment' in data:
            result.append(f"News Sentiment: {data['news_sentiment']}")
        
        return "\n".join(result)
    
    def format_ai_analysis(self, data: Dict) -> str:
        """Format AI analysis data for display"""
        if not data:
            return "No AI analysis available"
        
        result = []
        result.append(f"🤖 AI-Powered Analysis")
        result.append("-" * 50)
        
        if 'prediction' in data:
            result.append(f"AI Prediction: {data['prediction']}")
        
        if 'confidence' in data:
            result.append(f"AI Confidence: {data['confidence']}")
        
        if 'factors' in data:
            result.append("\nKey AI Factors:")
            for factor in data['factors'][:5]:
                result.append(f"  • {factor}")
        
        if 'recommendation' in data:
            result.append(f"\nAI Recommendation: {data['recommendation']}")
        
        return "\n".join(result)
    
    def show_price_info(self, symbol: str):
        """Show price and basic information"""
        try:
            logger.info(f"Fetching price information for {symbol}...")
            data = self.client.get_token_price(symbol)
            print("\n" + self.format_price_data(data))
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_trader_grade(self, symbol: str):
        """Show trader grade information"""
        try:
            logger.info(f"Fetching trader grade for {symbol}...")
            data = self.client.get_trader_grade(symbol)
            print("\n" + self.format_trader_grade(data))
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_trading_signals(self, symbol: str):
        """Show trading signals"""
        try:
            logger.info(f"Fetching trading signals for {symbol}...")
            data = self.client.get_trading_signals(symbol)
            print("\n" + self.format_trading_signals(data))
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_market_sentiment(self, symbol: str):
        """Show market sentiment"""
        try:
            logger.info(f"Fetching market sentiment for {symbol}...")
            data = self.client.get_market_sentiment(symbol)
            print("\n" + self.format_sentiment(data))
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_ai_analysis(self, symbol: str):
        """Show AI analysis"""
        try:
            logger.info(f"Fetching AI analysis for {symbol}...")
            data = self.client.get_ai_analysis(symbol)
            print("\n" + self.format_ai_analysis(data))
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_quantitative_metrics(self, symbol: str):
        """Show quantitative metrics"""
        try:
            logger.info(f"Fetching quantitative metrics for {symbol}...")
            data = self.client.get_quantitative_metrics(symbol)
            print(f"\n📊 Quantitative Metrics for {symbol}")
            print("-" * 50)
            print(json.dumps(data, indent=2, default=str))
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_resistance_support(self, symbol: str):
        """Show resistance and support levels"""
        try:
            logger.info(f"Fetching resistance/support for {symbol}...")
            data = self.client.get_resistance_support(symbol)
            print(f"\n📈 Resistance & Support Levels for {symbol}")
            print("-" * 50)
            print(json.dumps(data, indent=2, default=str))
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_scenario_analysis(self, symbol: str):
        """Show scenario analysis"""
        try:
            logger.info(f"Fetching scenario analysis for {symbol}...")
            data = self.client.get_scenario_analysis(symbol)
            print(f"\n🔮 Scenario Analysis for {symbol}")
            print("-" * 50)
            print(json.dumps(data, indent=2, default=str))
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_comprehensive_analysis(self, symbol: str):
        """Show comprehensive analysis"""
        try:
            logger.info(f"Fetching comprehensive analysis for {symbol}...")
            data = get_comprehensive_token_analysis(symbol)
            
            print(f"\n🔍 COMPREHENSIVE ANALYSIS FOR {symbol}")
            print("=" * 60)
            
            for analysis_type, analysis_data in data['data'].items():
                print(f"\n📋 {analysis_type.upper().replace('_', ' ')}")
                print("-" * 40)
                if analysis_data:
                    print(json.dumps(analysis_data, indent=2, default=str))
                else:
                    print("No data available")
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_multiple_tokens_analysis(self, symbol: str = None):
        """Show multiple tokens analysis"""
        try:
            symbols_input = self.get_user_input("Enter comma-separated list of symbols (e.g., BTC,ETH,ADA)")
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            if not symbols:
                logger.warning("No symbols provided for multiple analysis.")
                return

            logger.info(f"Fetching comprehensive analysis for multiple tokens: {', '.join(symbols)}...")
            data = get_multiple_tokens_analysis(symbols)
            
            print(f"\n🔍 COMPREHENSIVE ANALYSIS FOR MULTIPLE TOKENS")
            print("=" * 60)
            
            if 'data' in data and 'tokens' in data['data']:
                tokens_data = data['data']['tokens']
                for symbol in symbols:
                    print(f"\n📋 Analysis for {symbol}")
                    print("-" * 40)
                    # Find the token data for this symbol
                    token_data = None
                    if isinstance(tokens_data, list):
                        for token in tokens_data:
                            if token.get('symbol', '').upper() == symbol:
                                token_data = token
                                break
                    elif isinstance(tokens_data, dict):
                        token_data = tokens_data.get(symbol)
                    
                    if token_data:
                        print(json.dumps(token_data, indent=2, default=str))
                    else:
                        print(f"No data available for {symbol}")
            else:
                print("No comprehensive analysis data available")
            
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_top_tokens(self, symbol: str = None):
        """Show top tokens (symbol parameter ignored)"""
        try:
            limit = int(self.get_user_input("Enter number of top tokens to display", "10"))
            logger.info(f"Fetching top {limit} tokens...")
            data = self.client.get_top_tokens(limit=limit)
            
            print(f"\n🏆 TOP {limit} TOKENS")
            print("-" * 50)
            
            if 'tokens' in data:
                for i, token in enumerate(data['tokens'], 1):
                    print(f"{i:2d}. {token.get('symbol', 'Unknown')} - {token.get('name', 'Unknown')}")
            else:
                print(json.dumps(data, indent=2, default=str))
                
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def search_tokens(self, symbol: str = None):
        """Search for tokens (symbol parameter ignored)"""
        try:
            query = self.get_user_input("Enter search query")
            if not query:
                logger.warning("Search query cannot be empty")
                return
                
            logger.info(f"Searching for tokens matching '{query}'...")
            data = self.client.search_tokens(query)
            
            print(f"\n🔍 SEARCH RESULTS FOR '{query}'")
            print("-" * 50)
            
            if 'tokens' in data:
                for i, token in enumerate(data['tokens'], 1):
                    print(f"{i:2d}. {token.get('symbol', 'Unknown')} - {token.get('name', 'Unknown')}")
            else:
                print(json.dumps(data, indent=2, default=str))
                
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def show_performance_indices(self, symbol: str = None):
        """Show performance indices (symbol parameter ignored)"""
        try:
            logger.info("Fetching performance indices...")
            data = self.client.get_performance_indices()
            
            print(f"\n📊 PERFORMANCE INDICES")
            print("-" * 50)
            print(json.dumps(data, indent=2, default=str))
                
        except TokenMetricsAPIError as e:
            logger.error(f"API Error: {e.message}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    
    def exit_program(self, symbol: str = None):
        """Exit the program"""
        print("\n👋 Thank you for using Token Metrics API!")
        print("Goodbye! 🚀")
        sys.exit(0)
    
    def run(self):
        """Main run loop"""
        print("🚀 Welcome to Token Metrics API CLI!")
        
        # Initialize client
        if not self.initialize_client():
            logger.error("Failed to initialize API client. Exiting...")
            return
        
        # Test connection
        logger.info("Testing API connection...")
        if not test_api_connection():
            logger.error("API connection test failed. Please check your API key.")
            logger.info("Trying to test specific token functionality...")
            if not test_specific_token():
                logger.error("Specific token test also failed. Please verify your API key and internet connection.")
                return
        
        logger.success("API connection successful!")
        
        # Main loop
        while True:
            try:
                self.display_menu()
                choice = self.get_user_choice()
                
                # Get function and name
                func_name, func = self.available_modes[choice]
                
                # Check if function needs symbol parameter
                if choice in ['10', '11', '12', '13', '0']:
                    # These functions don't need symbol input
                    func()
                else:
                    # Get symbol for other functions
                    symbol = self.get_crypto_symbol()
                    func(symbol)
                
                # Wait for user to continue
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Program interrupted by user. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                input("\nPress Enter to continue...")

def main():
    """Main entry point"""
    try:
        cli = TokenMetricsCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 