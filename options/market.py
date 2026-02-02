"""
Options market with Limit Order Book for option contracts.
"""
import numpy as np
from dataclasses import dataclass
from core.lob import LimitOrderBook


@dataclass
class OptionContract:
    """Represents an option contract specification."""
    contract_id: int
    option_type: str  # "call" or "put"
    strike: float
    maturity: float  # Time to expiration
    underlying_price: float  # Current underlying price (for reference)
    
    def __post_init__(self):
        assert self.option_type in ["call", "put"], "option_type must be 'call' or 'put'"
        assert self.strike > 0, "strike must be positive"
        assert self.maturity > 0, "maturity must be positive"


class OptionsMarket:
    """Options market with LOB for each contract."""
    
    def __init__(self, tick_size, rng, contracts=None):
        """
        Args:
            tick_size: Minimum price increment for options
            rng: Random number generator
            contracts: List of OptionContract instances
        """
        self.tick_size = float(tick_size)
        self.rng = rng
        self.contracts = {}
        self.lobs = {}  # contract_id -> LimitOrderBook
        
        if contracts:
            for contract in contracts:
                self.add_contract(contract)

    def add_contract(self, contract: OptionContract):
        """Add a new option contract to the market."""
        self.contracts[contract.contract_id] = contract
        self.lobs[contract.contract_id] = LimitOrderBook(
            tick_size=self.tick_size,
            rng=self.rng
        )

    def get_contract(self, contract_id):
        """Get contract by ID."""
        return self.contracts.get(contract_id)

    def get_lob(self, contract_id):
        """Get LOB for a contract."""
        return self.lobs.get(contract_id)

    def place_limit(self, contract_id, agent_id, side, price, qty, ttl=1):
        """Place a limit order in the options market."""
        lob = self.lobs.get(contract_id)
        if lob is None:
            return None
        return lob.add_limit(agent_id=agent_id, side=side, price=price, qty=qty, ttl=ttl)

    def place_market(self, contract_id, agent_id, side, qty):
        """Place a market order in the options market."""
        lob = self.lobs.get(contract_id)
        if lob is None:
            return None
        return lob.add_market(agent_id=agent_id, side=side, qty=qty)

    def cancel(self, contract_id, order_id):
        """Cancel an order in the options market."""
        lob = self.lobs.get(contract_id)
        if lob is None:
            return False
        return lob.cancel(order_id)

    def get_mid_price(self, contract_id, fallback=None):
        """Get mid price for a contract."""
        lob = self.lobs.get(contract_id)
        if lob is None:
            return fallback
        return lob.mid_price(fallback if fallback is not None else 0.0)

    def get_spread(self, contract_id):
        """Get bid-ask spread for a contract."""
        lob = self.lobs.get(contract_id)
        if lob is None:
            return None
        return lob.spread()

    def get_depth(self, contract_id):
        """Get depth at best bid/ask for a contract."""
        lob = self.lobs.get(contract_id)
        if lob is None:
            return (0.0, 0.0)
        return lob.depth_at_best()

    def step_time(self):
        """Advance time for all LOBs (expire orders)."""
        for lob in self.lobs.values():
            lob.step_time()

    def reset_step_counters(self):
        """Reset step counters for all LOBs."""
        for lob in self.lobs.values():
            lob.reset_step_counters()

    def update_contract_maturities(self, dt):
        """Update time to maturity for all contracts and remove expired ones."""
        expired = []
        for contract_id, contract in list(self.contracts.items()):
            contract.maturity -= dt
            if contract.maturity <= 0:
                expired.append(contract_id)
        
        for contract_id in expired:
            del self.contracts[contract_id]
            del self.lobs[contract_id]
        
        return expired

