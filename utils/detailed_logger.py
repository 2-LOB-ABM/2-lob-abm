"""
Detailed logging system for simulation analysis.
Logs all important parameters at each step to CSV file for post-simulation analysis.
"""
import csv
import os
import sys
import time
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from options.pricing import PricingModel


class DetailedSimulationLogger:
    """
    Logs detailed simulation data to CSV file for analysis.
    Creates separate files for market data and dealer data.
    """
    
    def __init__(self, log_dir="simulation_logs", enable=True):
        self.enable = enable
        self.log_dir = log_dir
        self.market_file = None
        self.dealer_file = None
        self.market_writer = None
        self.dealer_writer = None
        self.market_fieldnames = None
        self.dealer_fieldnames = None
        
        if not self.enable:
            return
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamped filenames
        timestamp = int(time.time())
        self.market_filepath = os.path.join(self.log_dir, f"market_log_{timestamp}.csv")
        self.dealer_filepath = os.path.join(self.log_dir, f"dealers_log_{timestamp}.csv")
        
    def initialize(self, n_dealers=0):
        """Initialize CSV files with headers."""
        if not self.enable:
            return
        
        # Market log headers
        self.market_fieldnames = [
            'step', 'time',
            # Market state
            'price', 'volatility', 'regime', 'spread', 'depth_bid', 'depth_ask', 'imbalance',
            # Market activity
            'n_events', 'trades_count', 'volume', 'lambda_hawkes',
            # Option market (aggregate)
            'option_spread_avg', 'option_depth_avg',
            # Model distribution
            'model_dist_BS', 'model_dist_TFBS', 'model_dist_HESTON',
            # Average metrics across dealers
            'avg_reward_BS', 'avg_reward_TFBS', 'avg_reward_HESTON',
            'avg_quality_BS', 'avg_quality_TFBS', 'avg_quality_HESTON',
            'avg_hedge_error_BS', 'avg_hedge_error_TFBS', 'avg_hedge_error_HESTON',
        ]
        
        # Dealer log headers (one row per dealer per step)
        self.dealer_fieldnames = [
            'step', 'dealer_id', 'time',
            # Current strategy
            'current_strategy', 'strategy_switched',
            # Strategy weights
            'weight_BS', 'weight_TFBS', 'weight_HESTON',
            # Strategy quality
            'quality_BS', 'quality_TFBS', 'quality_HESTON',
            # Performance
            'reward', 'delta_pnl', 'wealth', 'last_wealth', 'cash',
            # Inventory
            'total_inventory', 'inventory_contracts',
            # Hedging
            'hedge_position', 'total_delta', 'last_delta',
            'hedge_error_current', 'hedge_error_BS', 'hedge_error_TFBS', 'hedge_error_HESTON',
            # Risk metrics
            'total_gamma', 'total_vega', 'risk_penalty', 'inventory_penalty',
            # Option positions (detailed)
            'position_call_ATM', 'position_put_ATM', 'position_put_OTM',
            # Trading activity
            'trades_received_count', 'quotes_placed',
        ]
        
        # Open market log file with buffering for performance
        self.market_file = open(self.market_filepath, 'w', newline='', buffering=8192)  # 8KB buffer
        self.market_writer = csv.DictWriter(self.market_file, fieldnames=self.market_fieldnames)
        self.market_writer.writeheader()
        
        # Open dealer log file with buffering for performance
        self.dealer_file = open(self.dealer_filepath, 'w', newline='', buffering=8192)  # 8KB buffer
        self.dealer_writer = csv.DictWriter(self.dealer_file, fieldnames=self.dealer_fieldnames)
        self.dealer_writer.writeheader()
        
        # Track steps for periodic flushing
        self.step_count = 0
        self.flush_interval = 100  # Flush every 100 steps
        self.log_every_n_steps = 1  # Log every N steps (1 = every step)
        
        print(f"Detailed logging initialized:")
        print(f"  Market log: {self.market_filepath}")
        print(f"  Dealers log: {self.dealer_filepath}")
    
    def log_step(self, model, step):
        """Log data for current simulation step."""
        if not self.enable:
            return
        
        # Optimize: log every N steps instead of every step
        self.step_count += 1
        if self.step_count % self.log_every_n_steps != 0:
            return
        
        current_time = step * model.dt
        
        # Collect market data
        market_data = self._collect_market_data(model, step, current_time)
        self.market_writer.writerow(market_data)
        
        # Collect dealer data
        dealers = [a for a in model.agents_list if hasattr(a, 'pricing_model')]
        for dealer in dealers:
            dealer_data = self._collect_dealer_data(dealer, model, step, current_time)
            self.dealer_writer.writerow(dealer_data)
        
        # Periodic flush to disk for performance (every N steps)
        if self.step_count % self.flush_interval == 0:
            self.market_file.flush()
            self.dealer_file.flush()
    
    def _collect_market_data(self, model, step, current_time):
        """Collect market-level data."""
        data = {
            'step': step,
            'time': current_time,
            'price': model.current_price,
            'volatility': model.volatility,
            'regime': model.regime,
            'spread': model.spread_log[-1] if model.spread_log else 0.0,
            'depth_bid': model.depth_bid_log[-1] if model.depth_bid_log else 0.0,
            'depth_ask': model.depth_ask_log[-1] if model.depth_ask_log else 0.0,
            'imbalance': model.imbalance_log[-1] if model.imbalance_log else 0.0,
            'n_events': model.n_events_log[-1] if model.n_events_log else 0,
            'trades_count': model.trade_count_log[-1] if model.trade_count_log else 0,
            'volume': model.volume_log[-1] if model.volume_log else 0.0,
            'lambda_hawkes': model.lambda_log[-1] if model.lambda_log else 0.0,
        }
        
        # Option market data
        if model.enable_options and model.option_contracts:
            spreads = []
            depths = []
            for contract in model.option_contracts:
                cid = contract.contract_id
                if cid in model.option_spreads_log and model.option_spreads_log[cid]:
                    spread = model.option_spreads_log[cid][-1]
                    if not (isinstance(spread, float) and (spread != spread or spread == float('inf'))):  # Not NaN or Inf
                        spreads.append(spread)
                if cid in model.option_depths_log and model.option_depths_log[cid]:
                    depth = model.option_depths_log[cid][-1]
                    if isinstance(depth, tuple) and len(depth) == 2:
                        depths.append(depth[0] + depth[1])
            
            data['option_spread_avg'] = sum(spreads) / len(spreads) if spreads else 0.0
            data['option_depth_avg'] = sum(depths) / len(depths) if depths else 0.0
        else:
            data['option_spread_avg'] = 0.0
            data['option_depth_avg'] = 0.0
        
        # Model distribution
        if model.enable_options and model.dealer_model_distribution_log:
            dist = model.dealer_model_distribution_log[-1]
            data['model_dist_BS'] = dist.get(PricingModel.BS, 0.0)
            data['model_dist_TFBS'] = dist.get(PricingModel.TFBS, 0.0)
            data['model_dist_HESTON'] = dist.get(PricingModel.HESTON, 0.0)
        else:
            data['model_dist_BS'] = 0.0
            data['model_dist_TFBS'] = 0.0
            data['model_dist_HESTON'] = 0.0
        
        # Average rewards by model
        if model.enable_options and model.dealer_reward_log:
            reward_dict = model.dealer_reward_log[-1]
            data['avg_reward_BS'] = reward_dict.get(PricingModel.BS, 0.0)
            data['avg_reward_TFBS'] = reward_dict.get(PricingModel.TFBS, 0.0)
            data['avg_reward_HESTON'] = reward_dict.get(PricingModel.HESTON, 0.0)
        else:
            data['avg_reward_BS'] = 0.0
            data['avg_reward_TFBS'] = 0.0
            data['avg_reward_HESTON'] = 0.0
        
        # Average quality by model
        if model.enable_options and model.dealer_strategy_quality_log:
            quality_dict = model.dealer_strategy_quality_log[-1]
            data['avg_quality_BS'] = quality_dict.get(PricingModel.BS, 0.0)
            data['avg_quality_TFBS'] = quality_dict.get(PricingModel.TFBS, 0.0)
            data['avg_quality_HESTON'] = quality_dict.get(PricingModel.HESTON, 0.0)
        else:
            data['avg_quality_BS'] = 0.0
            data['avg_quality_TFBS'] = 0.0
            data['avg_quality_HESTON'] = 0.0
        
        # Average hedge errors by model
        if model.enable_options and hasattr(model, 'dealer_hedge_errors_by_model'):
            data['avg_hedge_error_BS'] = (
                model.dealer_hedge_errors_by_model[PricingModel.BS][-1]
                if model.dealer_hedge_errors_by_model.get(PricingModel.BS) else 0.0
            )
            data['avg_hedge_error_TFBS'] = (
                model.dealer_hedge_errors_by_model[PricingModel.TFBS][-1]
                if model.dealer_hedge_errors_by_model.get(PricingModel.TFBS) else 0.0
            )
            data['avg_hedge_error_HESTON'] = (
                model.dealer_hedge_errors_by_model[PricingModel.HESTON][-1]
                if model.dealer_hedge_errors_by_model.get(PricingModel.HESTON) else 0.0
            )
        else:
            data['avg_hedge_error_BS'] = 0.0
            data['avg_hedge_error_TFBS'] = 0.0
            data['avg_hedge_error_HESTON'] = 0.0
        
        return data
    
    def _collect_dealer_data(self, dealer, model, step, current_time):
        """Collect dealer-specific data."""
        data = {
            'step': step,
            'dealer_id': dealer.unique_id,
            'time': current_time,
        }
        
        # Current strategy
        current_strategy = (
            dealer.current_strategy if hasattr(dealer, 'current_strategy') else dealer.pricing_model
        )
        data['current_strategy'] = current_strategy.value if hasattr(current_strategy, 'value') else str(current_strategy)
        data['strategy_switched'] = (
            1 if (hasattr(dealer, '_strategy_switched_this_step') and dealer._strategy_switched_this_step) else 0
        )
        
        # Strategy weights
        if hasattr(dealer, 'strategy_weights'):
            data['weight_BS'] = dealer.strategy_weights.get(PricingModel.BS, 0.0)
            data['weight_TFBS'] = dealer.strategy_weights.get(PricingModel.TFBS, 0.0)
            data['weight_HESTON'] = dealer.strategy_weights.get(PricingModel.HESTON, 0.0)
        else:
            data['weight_BS'] = 0.0
            data['weight_TFBS'] = 0.0
            data['weight_HESTON'] = 0.0
        
        # Strategy quality
        if hasattr(dealer, 'strategy_quality'):
            data['quality_BS'] = dealer.strategy_quality.get(PricingModel.BS, 0.0)
            data['quality_TFBS'] = dealer.strategy_quality.get(PricingModel.TFBS, 0.0)
            data['quality_HESTON'] = dealer.strategy_quality.get(PricingModel.HESTON, 0.0)
        else:
            data['quality_BS'] = 0.0
            data['quality_TFBS'] = 0.0
            data['quality_HESTON'] = 0.0
        
        # Performance
        data['reward'] = dealer.current_reward if hasattr(dealer, 'current_reward') else 0.0
        data['wealth'] = dealer.wealth if hasattr(dealer, 'wealth') else 0.0
        data['last_wealth'] = dealer.last_wealth if hasattr(dealer, 'last_wealth') else 0.0
        data['cash'] = dealer.cash if hasattr(dealer, 'cash') else 0.0
        data['delta_pnl'] = (
            (dealer.wealth - dealer.last_wealth) 
            if (hasattr(dealer, 'wealth') and hasattr(dealer, 'last_wealth') and dealer.last_wealth is not None)
            else 0.0
        )
        
        # Inventory
        total_inventory = sum(abs(inv_data.get("position", 0)) for inv_data in dealer.inventory.values())
        data['total_inventory'] = total_inventory
        data['inventory_contracts'] = len([inv for inv in dealer.inventory.values() if abs(inv.get("position", 0)) > 0.01])
        
        # Hedging
        data['hedge_position'] = dealer.hedge_position if hasattr(dealer, 'hedge_position') else 0.0
        data['total_delta'] = getattr(dealer, 'last_delta', 0.0)
        data['last_delta'] = getattr(dealer, 'last_delta', 0.0)
        
        # Hedge errors
        if hasattr(dealer, 'hedge_errors') and dealer.hedge_errors:
            data['hedge_error_current'] = dealer.hedge_errors[-1][1] if isinstance(dealer.hedge_errors[-1], tuple) else dealer.hedge_errors[-1]
        else:
            data['hedge_error_current'] = 0.0
        
        # Model-specific hedge errors
        if hasattr(dealer, 'model_hedge_errors'):
            bs_errors = dealer.model_hedge_errors.get(PricingModel.BS, [])
            tfbs_errors = dealer.model_hedge_errors.get(PricingModel.TFBS, [])
            heston_errors = dealer.model_hedge_errors.get(PricingModel.HESTON, [])
            
            data['hedge_error_BS'] = bs_errors[-1][1] if bs_errors and isinstance(bs_errors[-1], tuple) else 0.0
            data['hedge_error_TFBS'] = tfbs_errors[-1][1] if tfbs_errors and isinstance(tfbs_errors[-1], tuple) else 0.0
            data['hedge_error_HESTON'] = heston_errors[-1][1] if heston_errors and isinstance(heston_errors[-1], tuple) else 0.0
        else:
            data['hedge_error_BS'] = 0.0
            data['hedge_error_TFBS'] = 0.0
            data['hedge_error_HESTON'] = 0.0
        
        # Risk metrics (use cached values if available, otherwise skip to avoid slow calculation)
        # Skip expensive Greeks calculation during logging - use approximate values
        total_gamma = 0.0
        total_vega = 0.0
        # Only calculate if inventory is small (to avoid performance issues)
        if hasattr(dealer, 'inventory') and model.enable_options:
            total_inv = sum(abs(inv_data.get("position", 0)) for inv_data in dealer.inventory.values())
            if total_inv > 0.01 and total_inv < 100:  # Only calculate for reasonable inventory sizes
                for contract_id, inv_data in dealer.inventory.items():
                    contract = model.options_market.get_contract(contract_id)
                    if contract and abs(inv_data.get("position", 0)) > 0.01:
                        try:
                            greeks = dealer._get_greeks(
                                contract_id,
                                contract.option_type,
                                contract.strike,
                                contract.maturity
                            )
                            total_gamma += abs(inv_data.get("position", 0) * greeks.get("gamma", 0.0))
                            total_vega += abs(inv_data.get("position", 0) * greeks.get("vega", 0.0))
                        except:
                            pass
        
        data['total_gamma'] = total_gamma
        data['total_vega'] = total_vega
        data['risk_penalty'] = (
            dealer.kappa_gamma * total_gamma + dealer.kappa_vega * total_vega
            if hasattr(dealer, 'kappa_gamma') else 0.0
        )
        data['inventory_penalty'] = (
            dealer.kappa_inventory * (total_inventory ** 2)
            if hasattr(dealer, 'kappa_inventory') else 0.0
        )
        
        # Option positions by contract type
        position_call_atm = 0.0
        position_put_atm = 0.0
        position_put_otm = 0.0
        
        if model.enable_options and hasattr(dealer, 'inventory'):
            for contract_id, inv_data in dealer.inventory.items():
                contract = model.options_market.get_contract(contract_id)
                if contract:
                    position = inv_data.get("position", 0.0)
                    if contract.option_type == "call" and abs(contract.strike - model.current_price) < 1.0:
                        position_call_atm = position
                    elif contract.option_type == "put" and abs(contract.strike - model.current_price) < 1.0:
                        position_put_atm = position
                    elif contract.option_type == "put" and contract.strike < model.current_price * 0.95:
                        position_put_otm = position
        
        data['position_call_ATM'] = position_call_atm
        data['position_put_ATM'] = position_put_atm
        data['position_put_OTM'] = position_put_otm
        
        # Trading activity
        data['trades_received_count'] = len(dealer.trades_received) if hasattr(dealer, 'trades_received') else 0
        data['quotes_placed'] = sum(len(quotes) for quotes in dealer.live_quotes.values()) if hasattr(dealer, 'live_quotes') else 0
        
        return data
    
    def close(self):
        """Close log files."""
        if not self.enable:
            return
        
        # Final flush before closing
        if self.market_file:
            self.market_file.flush()
            self.market_file.close()
        if self.dealer_file:
            self.dealer_file.flush()
            self.dealer_file.close()
        
        print(f"Logging completed. Files saved:")
        print(f"  Market log: {self.market_filepath}")
        print(f"  Dealers log: {self.dealer_filepath}")
