import logging
from colorama import init, Fore, Back, Style
init(autoreset=True)

def setup_logging(module_name, log_level=logging.INFO):
    """
    Configure logging with comprehensive colored output for console.
    
    This function sets up logging with extensive color support using colorama,
    providing visual distinction for different log levels and enhanced readability.
    
    Args:
        module_name (str): Name of the module for logger identification
        log_level (int): Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Logger instance configured with colored formatting
    """
    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add comprehensive color support to logs
    class ColoredFormatter(logging.Formatter):
        """
        Custom formatter to add comprehensive colors and styling to log output.
        
        Supports all available colorama colors including foreground, background,
        and style modifications for maximum visual distinction.
        """
        # Enhanced color mapping with all available colors
        COLORS = {
            'CRITICAL': Fore.RED + Back.YELLOW + Style.BRIGHT,      # Red text, yellow background, bright
            'ERROR': Fore.RED + Style.BRIGHT,                      # Bright red
            'WARNING': Fore.YELLOW + Style.BRIGHT,                 # Bright yellow
            'INFO': Fore.CYAN + Style.NORMAL,                      # Cyan
            'DEBUG': Fore.BLUE + Style.DIM,                        # Dim blue
            'NOTSET': Fore.WHITE + Style.DIM,                      # Dim white
        }
        
        # Additional colors for custom use cases
        EXTENDED_COLORS = {
            'SUCCESS': Fore.GREEN + Style.BRIGHT,                  # Bright green
            'FAIL': Fore.RED + Back.WHITE + Style.BRIGHT,          # Red on white, bright
            'GPU': Fore.MAGENTA + Style.BRIGHT,                    # Bright magenta for GPU logs
            'CPU': Fore.BLUE + Style.NORMAL,                       # Blue for CPU logs
            'TRADE': Fore.GREEN + Back.BLACK + Style.BRIGHT,       # Green on black for trading
            'SIGNAL': Fore.YELLOW + Back.BLUE + Style.BRIGHT,      # Yellow on blue for signals
            'MODEL': Fore.CYAN + Back.BLACK + Style.BRIGHT,        # Cyan on black for models
            'DATA': Fore.WHITE + Style.BRIGHT,                     # Bright white for data
            'NETWORK': Fore.LIGHTBLUE_EX + Style.NORMAL,          # Light blue for network
            'MEMORY': Fore.LIGHTMAGENTA_EX + Style.NORMAL,        # Light magenta for memory
            'PERFORMANCE': Fore.LIGHTGREEN_EX + Style.BRIGHT,     # Light green for performance
            'ANALYSIS': Fore.LIGHTYELLOW_EX + Style.NORMAL,       # Light yellow for analysis
            'CONFIG': Fore.LIGHTCYAN_EX + Style.DIM,              # Light cyan for config
            'ALERT': Fore.RED + Back.YELLOW + Style.BRIGHT,       # Alert styling
            'PROCESS': Fore.LIGHTWHITE_EX + Style.NORMAL,         # Light white for processes
        }
        
        # Background colors for special emphasis
        BACKGROUND_STYLES = {
            'HIGHLIGHT': Back.YELLOW + Fore.BLACK + Style.BRIGHT,  # Black on yellow
            'EMPHASIS': Back.BLUE + Fore.WHITE + Style.BRIGHT,     # White on blue
            'URGENT': Back.RED + Fore.WHITE + Style.BRIGHT,        # White on red
            'SUCCESS_BG': Back.GREEN + Fore.WHITE + Style.BRIGHT,  # White on green
            'INFO_BG': Back.CYAN + Fore.BLACK + Style.BRIGHT,      # Black on cyan
        }

        def format(self, record):
            """
            Format log record with appropriate colors based on level and content.
            
            Args:
                record (LogRecord): The log record to format
                
            Returns:
                str: Formatted and colored log message
            """
            levelname = record.levelname
            message = str(record.msg)
            
            # Check for custom method prefixes first (higher priority)
            if message.startswith("SUCCESS:"):
                record.levelname = f"{self.EXTENDED_COLORS['SUCCESS']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['SUCCESS']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("GPU:"):
                record.levelname = f"{self.EXTENDED_COLORS['GPU']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['GPU']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("TRADE:"):
                record.levelname = f"{self.EXTENDED_COLORS['TRADE']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['TRADE']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("SIGNAL:"):
                record.levelname = f"{self.EXTENDED_COLORS['SIGNAL']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['SIGNAL']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("MODEL:"):
                record.levelname = f"{self.EXTENDED_COLORS['MODEL']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['MODEL']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("PERFORMANCE:"):
                record.levelname = f"{self.EXTENDED_COLORS['PERFORMANCE']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['PERFORMANCE']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("ANALYSIS:"):
                record.levelname = f"{self.EXTENDED_COLORS['ANALYSIS']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['ANALYSIS']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("DATA:"):
                record.levelname = f"{self.EXTENDED_COLORS['DATA']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['DATA']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("CONFIG:"):
                record.levelname = f"{self.EXTENDED_COLORS['CONFIG']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['CONFIG']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("NETWORK:"):
                record.levelname = f"{self.EXTENDED_COLORS['NETWORK']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['NETWORK']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("MEMORY:"):
                record.levelname = f"{self.EXTENDED_COLORS['MEMORY']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['MEMORY']}{message}{Style.RESET_ALL}"
                
            elif message.startswith("PROCESS:"):
                record.levelname = f"{self.EXTENDED_COLORS['PROCESS']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['PROCESS']}{message}{Style.RESET_ALL}"
            
            # Apply standard level colors
            elif levelname in self.COLORS:
                colored_levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
                record.levelname = colored_levelname
                
                # Color the message based on level
                colored_message = f"{self.COLORS[levelname]}{message}{Style.RESET_ALL}"
                record.msg = colored_message
            
            # Apply special colors based on message content keywords (fallback)
            elif any(keyword in message.upper() for keyword in ['FAILED', 'FAIL', 'CRASH', 'EXCEPTION']):
                record.levelname = f"{self.EXTENDED_COLORS['FAIL']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['FAIL']}{message}{Style.RESET_ALL}"
                
            elif any(keyword in message.upper() for keyword in ['ALERT', 'URGENT', 'CRITICAL']):
                record.levelname = f"{self.EXTENDED_COLORS['ALERT']}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.EXTENDED_COLORS['ALERT']}{message}{Style.RESET_ALL}"
            
            return super().format(record)

    # Apply colored formatter to console handler
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(ColoredFormatter(log_format))

    # Return logger for specified module
    logger = logging.getLogger(module_name)
    
    # Create a wrapper class to add custom methods without modifying Logger directly
    class EnhancedLogger:
        def __init__(self, base_logger):
            self._logger = base_logger
        
        def __getattr__(self, name):
            # Delegate all other attributes to the base logger
            return getattr(self._logger, name)
        
        def success(self, message, *args, **kwargs):
            """Log success messages with green color"""
            self._logger.info(f"SUCCESS: {message}", *args, **kwargs)
        
        def gpu(self, message, *args, **kwargs):
            """Log GPU-related messages with magenta color"""
            self._logger.info(f"GPU: {message}", *args, **kwargs)
        
        def trade(self, message, *args, **kwargs):
            """Log trading-related messages with special formatting"""
            self._logger.info(f"TRADE: {message}", *args, **kwargs)
        
        def signal(self, message, *args, **kwargs):
            """Log signal-related messages with special formatting"""
            self._logger.info(f"SIGNAL: {message}", *args, **kwargs)
        
        def model(self, message, *args, **kwargs):
            """Log model-related messages with special formatting"""
            self._logger.info(f"MODEL: {message}", *args, **kwargs)
        
        def performance(self, message, *args, **kwargs):
            """Log performance-related messages with special formatting"""
            self._logger.info(f"PERFORMANCE: {message}", *args, **kwargs)
        
        def analysis(self, message, *args, **kwargs):
            """Log analysis-related messages with special formatting"""
            self._logger.info(f"ANALYSIS: {message}", *args, **kwargs)
        
        def data(self, message, *args, **kwargs):
            """Log data-related messages with special formatting"""
            self._logger.info(f"DATA: {message}", *args, **kwargs)
        
        def config(self, message, *args, **kwargs):
            """Log configuration-related messages with special formatting"""
            self._logger.info(f"CONFIG: {message}", *args, **kwargs)
        
        def network(self, message, *args, **kwargs):
            """Log network-related messages with special formatting"""
            self._logger.info(f"NETWORK: {message}", *args, **kwargs)
        
        def memory(self, message, *args, **kwargs):
            """Log memory-related messages with special formatting"""
            self._logger.info(f"MEMORY: {message}", *args, **kwargs)
        
        def process(self, message, *args, **kwargs):
            """Log process-related messages with special formatting"""
            self._logger.info(f"PROCESS: {message}", *args, **kwargs)
    
    # Return the enhanced logger wrapper
    return EnhancedLogger(logger)

def get_all_available_colors():
    """
    Return a dictionary of all available colorama colors for reference.
    
    Returns:
        dict: Dictionary containing all colorama color constants
    """
    return {
        'FOREGROUND_COLORS': {
            'BLACK': Fore.BLACK,
            'RED': Fore.RED,
            'GREEN': Fore.GREEN,
            'YELLOW': Fore.YELLOW,
            'BLUE': Fore.BLUE,
            'MAGENTA': Fore.MAGENTA,
            'CYAN': Fore.CYAN,
            'WHITE': Fore.WHITE,
            'LIGHTBLACK_EX': Fore.LIGHTBLACK_EX,
            'LIGHTRED_EX': Fore.LIGHTRED_EX,
            'LIGHTGREEN_EX': Fore.LIGHTGREEN_EX,
            'LIGHTYELLOW_EX': Fore.LIGHTYELLOW_EX,
            'LIGHTBLUE_EX': Fore.LIGHTBLUE_EX,
            'LIGHTMAGENTA_EX': Fore.LIGHTMAGENTA_EX,
            'LIGHTCYAN_EX': Fore.LIGHTCYAN_EX,
            'LIGHTWHITE_EX': Fore.LIGHTWHITE_EX,
        },
        'BACKGROUND_COLORS': {
            'BLACK': Back.BLACK,
            'RED': Back.RED,
            'GREEN': Back.GREEN,
            'YELLOW': Back.YELLOW,
            'BLUE': Back.BLUE,
            'MAGENTA': Back.MAGENTA,
            'CYAN': Back.CYAN,
            'WHITE': Back.WHITE,
            'LIGHTBLACK_EX': Back.LIGHTBLACK_EX,
            'LIGHTRED_EX': Back.LIGHTRED_EX,
            'LIGHTGREEN_EX': Back.LIGHTGREEN_EX,
            'LIGHTYELLOW_EX': Back.LIGHTYELLOW_EX,
            'LIGHTBLUE_EX': Back.LIGHTBLUE_EX,
            'LIGHTMAGENTA_EX': Back.LIGHTMAGENTA_EX,
            'LIGHTCYAN_EX': Back.LIGHTCYAN_EX,
            'LIGHTWHITE_EX': Back.LIGHTWHITE_EX,
        },
        'STYLES': {
            'DIM': Style.DIM,
            'NORMAL': Style.NORMAL,
            'BRIGHT': Style.BRIGHT,
            'RESET_ALL': Style.RESET_ALL,
        }
    }

def print_color_demo():
    """
    Print a demonstration of all available colors and styles.
    Useful for testing and choosing color combinations.
    """
    colors = get_all_available_colors()
    
    print("\n" + "="*80)
    print("COLORAMA COLOR DEMONSTRATION")
    print("="*80)
    
    print("\nFOREGROUND COLORS:")
    for name, color in colors['FOREGROUND_COLORS'].items():
        print(f"{color}{name}: This is {name} text{Style.RESET_ALL}")
    
    print("\nBACKGROUND COLORS:")
    for name, color in colors['BACKGROUND_COLORS'].items():
        print(f"{color}{Fore.BLACK if name != 'BLACK' else Fore.WHITE}{name}: This is {name} background{Style.RESET_ALL}")
    
    print("\nSTYLES:")
    for name, style in colors['STYLES'].items():
        if name != 'RESET_ALL':
            print(f"{style}{Fore.CYAN}{name}: This is {name} style{Style.RESET_ALL}")
    
    print("\nCOMBINATIONS:")
    print(f"{Fore.RED + Back.YELLOW + Style.BRIGHT}CRITICAL: Red on Yellow, Bright{Style.RESET_ALL}")
    print(f"{Fore.GREEN + Back.BLACK + Style.BRIGHT}SUCCESS: Green on Black, Bright{Style.RESET_ALL}")
    print(f"{Fore.CYAN + Style.DIM}INFO: Cyan, Dim{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA + Style.BRIGHT}GPU: Magenta, Bright{Style.RESET_ALL}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    # Demo the colored logging
    print_color_demo()
    
    # Test the logger
    logger = setup_logging("test_module", logging.DEBUG)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test custom methods
    logger.success("Model training completed successfully!")
    logger.gpu("GPU acceleration enabled")
    logger.trade("BUY signal generated for BTCUSDT")
    logger.signal("LONG signal detected")
    logger.model("LSTM model loaded")
    logger.performance("Processing 1000 samples/second")
    logger.analysis("Analyzing market data...")
    logger.data("Loading data from source")
    logger.config("Configuration loaded")
    logger.network("Network connection established")
    logger.memory("Memory usage is within limits")
    logger.process("Process started with PID 1234")