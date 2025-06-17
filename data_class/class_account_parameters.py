from dataclasses import dataclass

@dataclass
class AccountParameters:
    """
    A class representing the parameters of a trading account.

    This class encapsulates all financial and operational parameters related to a trading account,
    including balance, transaction costs, and contract specifications.

    Attributes:
        initial_balance (float): Starting balance of the account in base currency units (default: 100000.0)
        spread (float): The difference between bid and ask price in decimal format (default: 0.0)
        slippage (float): Expected execution price slippage in decimal format (default: 0.0)
        contract_size (int): Standard lot size in base currency units (default: 100000)
        pip_size (float): The size of one pip in decimal format (default: 0.00001)
        pip_value (float): The value of one pip in base currency units, calculated as contract_size * pip_size
    """
    def __init__(self):
        self.initial_balance: float = 100000.0
        self.spread: float = 0.0
        self.slippage: float = 0.0
        self.contract_size: int = 100000
        self.pip_size: float = 0.0001
        self.pip_value: float = self.contract_size * self.pip_size
    
