"""
Option dealers with different pricing models and hedging strategies.
Includes model switching mechanism (H1).
"""
import numpy as np
from mesa import Agent
from collections import defaultdict
from options.pricing import PricingModel, get_pricer, BlackScholesPricer, TimeFractionalBSPricer, HestonPricer


class OptionDealer(Agent):
    """
    Option market maker that:
    1. Quotes options based on their pricing model
    2. Hedges positions in the underlying market
    3. Can switch models based on performance (H1)
    """
    
    def __init__(
        self,
        uid,
        model,
        pricing_model: PricingModel,
        contract_ids,
        r=0.0,                      # risk-free rate
        base_spread_pct=0.02,
        quote_size=10,
        hedge_frequency=1,          # Hedge every N steps (1 = every step)
        inventory_risk_aversion=0.1,
        enable_model_switching=True,
        switching_window=50,        # Steps to evaluate performance
        switching_threshold=0.05,   # Minimum improvement to switch
        **pricer_kwargs
    ):
        super().__init__(model)
        self.pricing_model = pricing_model
        self.contract_ids = list(contract_ids)
        self.r = float(r)
        self.base_spread_pct = float(base_spread_pct)
        self.quote_size = int(quote_size)
        self.hedge_frequency = int(hedge_frequency)
        self.inventory_risk_aversion = float(inventory_risk_aversion)
        self.enable_model_switching = bool(enable_model_switching)
        self.switching_window = int(switching_window)
        self.switching_threshold = float(switching_threshold)
        

        self.pricer = get_pricer(pricing_model, **pricer_kwargs)
        
        # Inventory tracking: contract_id -> (position, avg_price)
        self.inventory = defaultdict(lambda: {"position": 0, "avg_price": 0.0})
        
        # Hedging: delta position in underlying
        self.hedge_position = 0.0  
        
        # Performance tracking for model switching
        self.hedge_errors = []          # (step, error) 
        self.pnl_history = [] 
        self.total_pnl = 0.0  
        self.last_hedge_step = 0
        self.last_hedge_price = None    # Price when last hedge was executed
        self.last_option_value = 0.0    # Last calculated option portfolio value
        self.last_delta = 0.0 
        
        # Track when we get new option positions (trades executed against our quotes)
        self.trades_received = []  
        
        self.last_switch_step = -1000  # Step when we last switched models
        self.switch_cooldown = 20      # Minimum steps between switches
        
        # Track error trend for trend-based switching
        self.error_trend = [] 
        self.trend_window = 5          # Number of steps to check for increasing trend
        
        # Performance optimization: calculate all-model errors less frequently
        self._last_all_models_error_calc = -10
        self._all_models_error_frequency = 5  # Calculate every N steps
        
        self.live_quotes = defaultdict(list)  
        
        self.available_models = [PricingModel.BS, PricingModel.TFBS, PricingModel.HESTON]
        
        # Replicator switching ======================================================
        n_strategies = len(self.available_models)
        self.strategy_weights = {m: 1.0 / n_strategies for m in self.available_models}
        self.strategy_quality = {m: 0.0 for m in self.available_models}
         
        self.alpha = 0.05          # smoothing parameter for Q_s (reduced from 0.1 for slower forgetting)
        self.eta = 1.5             # replicator dynamics parameter (increased from 0.5 for faster adaptation)
        self.epsilon_floor = 0.01  # minimum weight floor to allow strategies to return
        
        self.switch_cost = 0.01  
        self.min_hold_time = 1  # Reduced from 5 to allow more frequent switching
        self.last_strategy_switch_step = -1000  
        self.current_strategy = pricing_model  
        
        # Track hedging errors for ALL models ========================================
        self.model_hedge_errors = {m: [] for m in self.available_models}
        self.model_pricers = {}
        # Store last delta and option value for each model to calculate real hedging errors
        self.model_last_delta = {m: 0.0 for m in self.available_models}
        self.model_last_option_value = {m: 0.0 for m in self.available_models}
        for model_type in self.available_models:
            if model_type == PricingModel.TFBS:
                self.model_pricers[model_type] = get_pricer(model_type, alpha=0.85, n_mc=1000)
            elif model_type == PricingModel.HESTON:
                self.model_pricers[model_type] = get_pricer(model_type, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, n_mc=1000)
            else:
                self.model_pricers[model_type] = get_pricer(model_type)
        
        self.wealth = 0.0       # total wealth (options + hedge + cash)
        self.last_wealth = None  # previous step wealth (None = not initialized yet)
        self.cash = 0.0         # cash position
        self._first_reward_calc = True  # Flag for first reward calculation
        
        # Reward function parameters
        self.lambda_risk = 0.1      # Risk penalty coefficient
        self.kappa_gamma = 0.001    # Gamma risk penalty coefficient (reduced from 0.01)
        self.kappa_vega = 0.001     # Vega risk penalty coefficient (reduced from 0.01)
        self.kappa_inventory = 0.001  # Inventory penalty coefficient
        
        # Track reward for logging
        self.current_reward = 0.0
        
    def _get_reservation_price(self, contract_id, option_type, strike, maturity):
        """Calculate reservation price based on current model."""
        S = float(self.model.current_price)
        sigma = float(self.model.volatility)
        
        if self.pricing_model == PricingModel.BS:
            if option_type == "call":
                price = self.pricer.price_call(S, strike, self.r, sigma, maturity)
            else:
                price = self.pricer.price_put(S, strike, self.r, sigma, maturity)
        elif self.pricing_model == PricingModel.TFBS:
            seed = self.model.rng.integers(0, 2**31)
            if option_type == "call":
                price = self.pricer.price_call(S, strike, self.r, sigma, maturity, seed=seed)
            else:
                price = self.pricer.price_put(S, strike, self.r, sigma, maturity, seed=seed)
        elif self.pricing_model == PricingModel.HESTON:
            seed = self.model.rng.integers(0, 2**31)
            if option_type == "call":
                price = self.pricer.price_call(S, strike, self.r, sigma, maturity, seed=seed)
            else:
                price = self.pricer.price_put(S, strike, self.r, sigma, maturity, seed=seed)
        else:
            raise ValueError(f"Unknown pricing model: {self.pricing_model}")
        
        # adjust for inventory risk
        inv = self.inventory[contract_id]["position"]
        inventory_adjustment = self.inventory_risk_aversion * inv * price * 0.01

        return float(price - inventory_adjustment)
    
    def _get_greeks(self, contract_id, option_type, strike, maturity):
        """Calculate Greeks for hedging."""
        S = float(self.model.current_price)
        sigma = float(max(self.model.volatility, 1e-6))  # Ensure sigma > 0
        
        if self.pricing_model == PricingModel.BS:
            if option_type == "call":
                delta = self.pricer.delta_call(S, strike, self.r, sigma, maturity)
                gamma = self.pricer.gamma(S, strike, self.r, sigma, maturity)
                vega = self.pricer.vega(S, strike, self.r, sigma, maturity)
            else:
                delta = self.pricer.delta_put(S, strike, self.r, sigma, maturity)
                gamma = self.pricer.gamma(S, strike, self.r, sigma, maturity)
                vega = self.pricer.vega(S, strike, self.r, sigma, maturity)
        elif self.pricing_model == PricingModel.TFBS:
            seed = self.model.rng.integers(0, 2**31)
            if option_type == "call":
                delta = self.pricer.delta_call(S, strike, self.r, sigma, maturity, seed=seed)
                gamma = self.pricer.gamma(S, strike, self.r, sigma, maturity, seed=seed)
                vega = self.pricer.vega(S, strike, self.r, sigma, maturity, seed=seed)
            else:
                delta = self.pricer.delta_put(S, strike, self.r, sigma, maturity, seed=seed)
                gamma = self.pricer.gamma(S, strike, self.r, sigma, maturity, seed=seed)
                vega = self.pricer.vega(S, strike, self.r, sigma, maturity, seed=seed)
        elif self.pricing_model == PricingModel.HESTON:
            seed = self.model.rng.integers(0, 2**31)
            if option_type == "call":
                delta = self.pricer.delta_call(S, strike, self.r, sigma, maturity, seed=seed)
                gamma = self.pricer.gamma(S, strike, self.r, sigma, maturity, seed=seed)
                vega = self.pricer.vega(S, strike, self.r, sigma, maturity, seed=seed)
            else:
                delta = self.pricer.delta_put(S, strike, self.r, sigma, maturity, seed=seed)
                gamma = self.pricer.gamma(S, strike, self.r, sigma, maturity, seed=seed)
                vega = self.pricer.vega(S, strike, self.r, sigma, maturity, seed=seed)
        else:
            raise ValueError(f"Unknown pricing model: {self.pricing_model}")
        
        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}
    
    def _calculate_model_error(self, model_type, price_change):
        """
        Calculate REAL hedging error for this model using actual positions.
        This computes what the error would have been if we had used this model's delta.
        Now includes gamma risk to make errors more distinct between models.
        """
        if model_type not in self.model_pricers:
            return 0.0
        
        if abs(price_change) < 1e-6:
            return 0.0
        
        # If we have no inventory, return small baseline error
        total_inventory = sum(abs(inv_data["position"]) for inv_data in self.inventory.values())
        if total_inventory < 0.01:
            return 0.0
        
        S = float(self.model.current_price)
        sigma = float(max(self.model.volatility, 1e-6))
        pricer = self.model_pricers[model_type]
        
        # Calculate current option portfolio value using THIS model
        current_option_value = 0.0
        current_total_delta = 0.0
        current_total_gamma = 0.0
        
        for contract_id, inv_data in self.inventory.items():
            contract = self.model.options_market.get_contract(contract_id)
            if contract is None:
                continue
            
            position = inv_data["position"]
            if abs(position) < 0.01:
                continue
            
            # Calculate option price, delta, and gamma using this model
            if model_type == PricingModel.BS:
                if contract.option_type == "call":
                    option_price = pricer.price_call(S, contract.strike, self.r, sigma, contract.maturity)
                    delta = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity)
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity)
                else:
                    option_price = pricer.price_put(S, contract.strike, self.r, sigma, contract.maturity)
                    delta = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity)
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity)
            elif model_type == PricingModel.TFBS:
                seed = self.model.rng.integers(0, 2**31)
                if contract.option_type == "call":
                    option_price = pricer.price_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    delta = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                else:
                    option_price = pricer.price_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    delta = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
            elif model_type == PricingModel.HESTON:
                seed = self.model.rng.integers(0, 2**31)
                if contract.option_type == "call":
                    option_price = pricer.price_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    delta = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                else:
                    option_price = pricer.price_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    delta = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
            else:
                option_price = 0.0
                delta = 0.0
                gamma = 0.0
            
            current_option_value += position * option_price
            current_total_delta += position * delta
            current_total_gamma += position * gamma
        
        # Calculate what the hedge PnL would have been with this model's delta
        # We use the last delta we stored for this model (from previous hedge)
        last_delta_for_model = self.model_last_delta[model_type]
        hedge_pnl_if_used_this_model = last_delta_for_model * price_change
        
        # Calculate option PnL using this model's pricing
        last_option_value_for_model = self.model_last_option_value[model_type]
        option_pnl = current_option_value - last_option_value_for_model
        
        # Real hedging error = |option_pnl - hedge_pnl|
        # This is what the error would have been if we had used this model
        real_error = abs(option_pnl - hedge_pnl_if_used_this_model)
        
        # Add gamma risk component to make errors more distinct between models
        # Gamma risk = 0.5 * gamma * price_change^2 (second-order effect)
        gamma_risk = 0.5 * abs(current_total_gamma) * (price_change ** 2)
        total_error = real_error + gamma_risk
        
        # Normalize by price change to make errors comparable
        if abs(price_change) > 1e-6:
            normalized_error = total_error / abs(price_change)
        else:
            normalized_error = total_error
        
        # Update stored values for next iteration
        self.model_last_delta[model_type] = current_total_delta
        self.model_last_option_value[model_type] = current_option_value
        
        return normalized_error
    
    def _check_option_trades(self):
        """Check if any of our option quotes were executed by monitoring LOB trades."""
        if not hasattr(self.model, 'options_market'):
            return
        
        current_step = self.model.market.book.t
        
        # Track which trades we've already processed to avoid double-counting
        if not hasattr(self, '_processed_trades'):
            self._processed_trades = set()
        
        # Check each contract's LOB for trades involving our orders
        for contract_id in self.contract_ids:
            lob = self.model.options_market.get_lob(contract_id)
            if lob is None:
                continue
            
            # Check ALL trades in this contract's LOB (not just recent ones)
            # Use trade tuple as unique identifier to avoid double processing
            for trade in lob.trades:
                step, price, qty, passive_id, aggressive_id = trade
                trade_key = (contract_id, step, price, qty, passive_id, aggressive_id)
                
                # Skip if we already processed this trade
                if trade_key in self._processed_trades:
                    continue
                
                # If we were the passive side (our quote was hit)
                if passive_id == self.unique_id:
                    # Determine side: if our ask was hit, we sold; if our bid was hit, we bought
                    # Check our live quotes to see which side was hit
                    side = None
                    for oid in self.live_quotes.get(contract_id, []):
                        order = lob.orders.get(oid)
                        if order is None:
                            continue
                        # Check if this order was at the trade price (with tolerance)
                        order_price = order.price_ticks * lob.tick_size if order.price_ticks else None
                        if order_price and abs(order_price - price) < 0.01:  # Increased tolerance
                            # If order was on ask side (sell), we sold; if bid side (buy), we bought
                            side = order.side
                            break
                    
                    # If we couldn't determine side from live quotes, infer from trade
                    # If we were passive, we provided liquidity - check if trade was at bid or ask
                    if side is None:
                        # Try to infer from best bid/ask at time of trade
                        # If price is closer to bid, we likely sold (ask was hit)
                        # If price is closer to ask, we likely bought (bid was hit)
                        best_bid = lob.best_bid()
                        best_ask = lob.best_ask()
                        if best_bid is not None and best_ask is not None:
                            mid = (best_bid + best_ask) / 2
                            if price >= mid:
                                side = "sell"  # Our ask was hit
                            else:
                                side = "buy"   # Our bid was hit
                    
                    if side:
                        # Record the trade
                        self._record_trade(contract_id, side, qty, price)
                        self.trades_received.append((step, contract_id, side, qty, price))
                        self._processed_trades.add(trade_key)
                
                # Also check if we were the aggressive side (we hit someone's quote)
                elif aggressive_id == self.unique_id:
                    # If we were aggressive, we hit someone's quote
                    # This is less common for market makers, but we should track it
                    # Determine side: if we bought, we hit an ask; if we sold, we hit a bid
                    # For now, we'll infer from our intent (but this should be tracked separately)
                    # Skip aggressive trades for now as dealers are primarily passive
                    pass
    
    def _hedge_position(self):
        """Execute delta hedge in underlying market."""
        if not hasattr(self.model, 'options_market'):
            return
        
        # Calculate current option portfolio value
        current_option_value = 0.0
        total_delta = 0.0
        
        for contract_id, inv_data in self.inventory.items():
            contract = self.model.options_market.get_contract(contract_id)
            if contract is None:
                continue
            
            # Calculate option value using current model
            res_price = self._get_reservation_price(
                contract_id,
                contract.option_type,
                contract.strike,
                contract.maturity
            )
            current_option_value += inv_data["position"] * res_price
            
            # Calculate delta
            greeks = self._get_greeks(
                contract_id,
                contract.option_type,
                contract.strike,
                contract.maturity
            )
            
            # Delta exposure = position * delta
            total_delta += inv_data["position"] * greeks["delta"]
        
        # Calculate hedge needed
        delta_to_hedge = total_delta - self.hedge_position
        current_price = float(self.model.current_price)
        
        # Initialize model tracking values on first hedge
        if self.last_hedge_price is None:
            # First hedge - initialize all model tracking
            for model_type in self.available_models:
                # Calculate initial option value and delta for each model
                S = float(self.model.current_price)
                sigma = float(max(self.model.volatility, 1e-6))
                pricer = self.model_pricers[model_type]
                
                model_option_value = 0.0
                model_delta = 0.0
                
                for contract_id, inv_data in self.inventory.items():
                    contract = self.model.options_market.get_contract(contract_id)
                    if contract is None:
                        continue
                    
                    position = inv_data["position"]
                    if abs(position) < 0.01:
                        continue
                    
                    if model_type == PricingModel.BS:
                        if contract.option_type == "call":
                            option_price = pricer.price_call(S, contract.strike, self.r, sigma, contract.maturity)
                            delta = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity)
                        else:
                            option_price = pricer.price_put(S, contract.strike, self.r, sigma, contract.maturity)
                            delta = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity)
                    elif model_type == PricingModel.TFBS:
                        seed = self.model.rng.integers(0, 2**31)
                        if contract.option_type == "call":
                            option_price = pricer.price_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                            delta = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                        else:
                            option_price = pricer.price_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                            delta = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    elif model_type == PricingModel.HESTON:
                        seed = self.model.rng.integers(0, 2**31)
                        if contract.option_type == "call":
                            option_price = pricer.price_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                            delta = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                        else:
                            option_price = pricer.price_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                            delta = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    else:
                        option_price = 0.0
                        delta = 0.0
                    
                    model_option_value += position * option_price
                    model_delta += position * delta
                
                self.model_last_option_value[model_type] = model_option_value
                self.model_last_delta[model_type] = model_delta
        
        # Track hedging error for CURRENT model and ALL other models
        # This allows us to compare models and choose the best one
        if self.last_hedge_price is not None and self.last_hedge_step > 0:
            price_change = current_price - self.last_hedge_price
            
            # Calculate error for current model
            # Always calculate error, even if inventory is small, for better tracking
            if abs(self.last_option_value) > 1e-6 or abs(current_option_value) > 1e-6 or abs(price_change) > 1e-6:
                # We have inventory - track actual hedging error
                hedge_pnl = self.hedge_position * price_change
                option_pnl = current_option_value - self.last_option_value
                hedge_error = option_pnl - hedge_pnl
                
                # Delta error: how much did delta change vs what we hedged?
                # This captures gamma risk (delta changes with price)
                delta_error = abs(total_delta - self.last_delta) * abs(price_change) * 0.1
                
                # Total error is the absolute hedging error
                # Normalize by price change to make it comparable across different price movements
                if abs(price_change) > 1e-6:
                    normalized_error = abs(hedge_error) / abs(price_change) if abs(price_change) > 1e-6 else abs(hedge_error)
                else:
                    normalized_error = abs(hedge_error)
                
                total_error = normalized_error + delta_error
                
                self.hedge_errors.append((self.model.market.book.t, total_error))
                self.pnl_history.append(option_pnl - hedge_pnl)
                self.total_pnl += (option_pnl - hedge_pnl)
                
                # Track error trend
                self.error_trend.append(total_error)
                if len(self.error_trend) > self.trend_window:
                    self.error_trend.pop(0)  # Keep only recent errors
            
            # Calculate errors for ALL models to compare performance
            # Optimize: only calculate every N steps to reduce computation
            current_step = self.model.market.book.t
            steps_since_last_calc = current_step - self._last_all_models_error_calc
            
            if steps_since_last_calc >= self._all_models_error_frequency:
                total_inventory = sum(abs(inv_data["position"]) for inv_data in self.inventory.values())
                
                for model_type in self.available_models:
                    model_error = self._calculate_model_error(model_type, price_change)
                    if abs(price_change) > 1e-6 or total_inventory > 0.01:
                        self.model_hedge_errors[model_type].append((current_step, model_error))
                        if len(self.model_hedge_errors[model_type]) > self.switching_window * 2:
                            self.model_hedge_errors[model_type] = self.model_hedge_errors[model_type][-self.switching_window:]
                
                self._last_all_models_error_calc = current_step
        
        # Execute hedge
        # Use smaller threshold for more precise hedging, especially important for high gamma positions
        hedge_threshold = 0.001  # Reduced from 0.01 for better hedging precision
        if abs(delta_to_hedge) > hedge_threshold:
            # Execute hedge as market order
            qty = int(abs(delta_to_hedge))
            side = "buy" if delta_to_hedge > 0 else "sell"
            
            # Get execution price (approximate using current mid price)
            # In reality, market orders execute at best bid/ask, but for simplicity use mid
            execution_price = current_price
            self.model.market.place_market(self.unique_id, side, qty)
            
            # Update cash: buying costs money, selling receives money
            if side == "buy":
                self.cash -= qty * execution_price
            else:  # sell
                self.cash += qty * execution_price
            
            self.hedge_position += delta_to_hedge
        
        # Update tracking
        self.last_hedge_step = self.model.market.book.t
        self.last_hedge_price = current_price
        self.last_option_value = current_option_value
        self.last_delta = total_delta
    
    def _update_quotes(self):
        """Update option quotes based on reservation prices."""
        if not hasattr(self.model, 'options_market'):
            return
        
        for contract_id in self.contract_ids:
            contract = self.model.options_market.get_contract(contract_id)
            if contract is None:
                continue
            
            # Calculate reservation price
            res_price = self._get_reservation_price(
                contract_id,
                contract.option_type,
                contract.strike,
                contract.maturity
            )
            
            # Set bid/ask around reservation price
            spread = res_price * self.base_spread_pct
            bid_price = res_price - spread / 2
            ask_price = res_price + spread / 2
            
            # Cancel old quotes
            for oid in self.live_quotes[contract_id]:
                self.model.options_market.cancel(contract_id, oid)
            self.live_quotes[contract_id] = []
            
            # Place new quotes
            bid_oid = self.model.options_market.place_limit(
                contract_id, self.unique_id, "buy", bid_price, self.quote_size, ttl=10
            )
            ask_oid = self.model.options_market.place_limit(
                contract_id, self.unique_id, "sell", ask_price, self.quote_size, ttl=10
            )
            
            if bid_oid is not None:
                self.live_quotes[contract_id].append(bid_oid)
            if ask_oid is not None:
                self.live_quotes[contract_id].append(ask_oid)
    
    def _record_trade(self, contract_id, side, qty, price):
        """Record an option trade and update inventory."""
        position_change = qty if side == "buy" else -qty
        old_position = self.inventory[contract_id]["position"]
        old_avg = self.inventory[contract_id]["avg_price"]
        
        new_position = old_position + position_change
        
        if new_position == 0:
            self.inventory[contract_id] = {"position": 0, "avg_price": 0.0}
        else:
            # Update average price
            if old_position == 0:
                new_avg = price
            elif position_change > 0:  # Adding to position
                new_avg = (old_position * old_avg + position_change * price) / new_position
            else:  # Reducing position
                new_avg = old_avg  # Keep same average
            
            self.inventory[contract_id] = {"position": new_position, "avg_price": new_avg}
        
        # Update cash: if we sold (side="sell"), we receive cash; if we bought, we pay cash
        if side == "sell":
            self.cash += qty * price
        else:  # buy
            self.cash -= qty * price
    
    def _select_strategy(self):
        """
        Select strategy probabilistically based on weights (Вариант 1 из PDF).
        Returns the selected PricingModel.
        """
        if not self.enable_model_switching:
            return self.pricing_model
        
        # Check minimum hold time
        current_step = self.model.market.book.t
        steps_since_switch = current_step - self.last_strategy_switch_step
        
        if steps_since_switch < self.min_hold_time:
            # Keep current strategy
            return self.current_strategy
        
        # Select strategy based on weights (categorical distribution)
        weights_list = [self.strategy_weights[m] for m in self.available_models]
        selected_idx = self.model.rng.choice(len(self.available_models), p=weights_list)
        selected_strategy = self.available_models[selected_idx]
        
        # Update switch step if strategy changed
        if selected_strategy != self.current_strategy:
            self.last_strategy_switch_step = current_step
        
        return selected_strategy
    
    def _calculate_reward(self):
        """
        Calculate reward for the current step.
        r(t) = ΔP&L(t) - λ·Risk(t) - κ·InventoryPenalty(t)
        
        Returns reward value.
        """
        # Calculate current wealth (mark-to-market)
        # Wealth = option portfolio value + hedge position value + cash
        
        # Option portfolio value
        option_value = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        
        for contract_id, inv_data in self.inventory.items():
            contract = self.model.options_market.get_contract(contract_id)
            if contract is None:
                continue
            
            # Calculate option value using current strategy
            res_price = self._get_reservation_price(
                contract_id,
                contract.option_type,
                contract.strike,
                contract.maturity
            )
            option_value += inv_data["position"] * res_price
            
            # Calculate Greeks for risk calculation
            greeks = self._get_greeks(
                contract_id,
                contract.option_type,
                contract.strike,
                contract.maturity
            )
            total_gamma += abs(inv_data["position"] * greeks["gamma"])
            total_vega += abs(inv_data["position"] * greeks["vega"])
        
        # Hedge position value (mark-to-market)
        current_price = float(self.model.current_price)
        hedge_value = self.hedge_position * current_price
        
        # Total wealth
        self.wealth = option_value + hedge_value + self.cash
        
        # ΔP&L = change in wealth
        # On first step, initialize last_wealth to current wealth to avoid artificial jump
        if self.last_wealth is None or self._first_reward_calc:
            self.last_wealth = self.wealth
            self._first_reward_calc = False
            delta_pnl = 0.0  # No P&L on first step
        else:
            delta_pnl = self.wealth - self.last_wealth
        
        # Smooth delta_pnl to reduce volatility (exponential moving average)
        if not hasattr(self, '_smoothed_pnl'):
            self._smoothed_pnl = delta_pnl
        else:
            # Use EMA with alpha=0.3 to smooth out short-term volatility
            self._smoothed_pnl = 0.3 * delta_pnl + 0.7 * self._smoothed_pnl
        
        # Use smoothed PnL for reward calculation to make it more stable
        smoothed_delta_pnl = self._smoothed_pnl
        
        # Risk penalty: gamma and vega exposure (using linear penalty instead of quadratic for more realistic values)
        risk_penalty = self.kappa_gamma * total_gamma + self.kappa_vega * total_vega
        
        # Inventory penalty: squared inventory
        total_inventory = sum(abs(inv_data["position"]) for inv_data in self.inventory.values())
        inventory_penalty = self.kappa_inventory * (total_inventory ** 2)
        
        # Calculate reward using smoothed PnL
        reward = smoothed_delta_pnl - self.lambda_risk * risk_penalty - inventory_penalty
        
        # Apply switching cost if strategy changed
        if hasattr(self, '_strategy_switched_this_step') and self._strategy_switched_this_step:
            reward -= self.switch_cost
        
        # Update last wealth for next step
        self.last_wealth = self.wealth
        
        return reward
    
    def _update_strategy_quality(self, selected_strategy, reward):
        """
        Update quality score Q_s for the selected strategy using exponential smoothing.
        Q_s(t+1) = (1-α) * Q_s(t) + α * r(t)
        
        Now also directly incorporates hedging errors into quality calculation
        to make model selection more accurate.
        """
        if not self.enable_model_switching:
            return
        
        # Calculate hedging error penalty for selected strategy
        hedge_error_penalty = 0.0
        if hasattr(self, 'model_hedge_errors') and len(self.model_hedge_errors.get(selected_strategy, [])) > 0:
            recent_errors = [e[1] for e in self.model_hedge_errors.get(selected_strategy, [])[-10:]]
            if recent_errors:
                avg_error = np.mean(recent_errors)
                # Penalize high hedging errors (scale: 1.0 error = -0.1 reward penalty)
                hedge_error_penalty = -avg_error * 0.1
        
        # Adjusted reward = reward - hedging error penalty
        adjusted_reward = reward + hedge_error_penalty
        
        # Update selected strategy with full weight
        current_q = self.strategy_quality[selected_strategy]
        new_q = (1.0 - self.alpha) * current_q + self.alpha * adjusted_reward
        self.strategy_quality[selected_strategy] = new_q
        
        # Update other strategies with counterfactual reward estimation
        # Use hedging errors to estimate what reward would have been with other strategies
        off_policy_alpha = self.alpha * 0.2
        
        for strategy in self.available_models:
            if strategy != selected_strategy:
                current_q_other = self.strategy_quality[strategy]
                
                # Estimate counterfactual reward based on hedging errors
                # If this strategy has lower hedging errors, it would have better reward
                if hasattr(self, 'model_hedge_errors') and len(self.model_hedge_errors.get(strategy, [])) > 0:
                    # Get recent errors for both strategies
                    recent_errors_selected = [e[1] for e in self.model_hedge_errors.get(selected_strategy, [])[-10:]]
                    recent_errors_other = [e[1] for e in self.model_hedge_errors.get(strategy, [])[-10:]]
                    
                    if recent_errors_selected and recent_errors_other:
                        avg_error_selected = np.mean(recent_errors_selected)
                        avg_error_other = np.mean(recent_errors_other)
                        
                        # Estimate reward difference based on error difference
                        # Lower error -> higher reward (inverse relationship)
                        error_diff = avg_error_selected - avg_error_other
                        # Scale error difference to reward scale
                        estimated_reward_diff = -error_diff * 0.1  # Same scale as penalty above
                        counterfactual_reward = adjusted_reward + estimated_reward_diff
                    else:
                        counterfactual_reward = adjusted_reward * 0.5
                else:
                    # No error data - use conservative estimate
                    counterfactual_reward = adjusted_reward * 0.5
                
                # Update quality with counterfactual reward
                new_q_other = (1.0 - off_policy_alpha) * current_q_other + off_policy_alpha * counterfactual_reward
                self.strategy_quality[strategy] = new_q_other
    
    def _update_strategy_weights(self):
        """
        Update strategy weights using replicator dynamics.
        w_s(t+1) = w_s(t) * exp(η * Q_s(t+1)) / Σ_j w_j(t) * exp(η * Q_j(t+1))
        
        Also applies epsilon-floor to prevent weights from going to zero.
        """
        if not self.enable_model_switching:
            return
        
        # Calculate new unnormalized weights
        new_weights = {}
        denominator = 0.0
        
        for strategy in self.available_models:
            # Apply epsilon-floor
            current_weight = max(self.strategy_weights[strategy], self.epsilon_floor)
            quality = self.strategy_quality[strategy]
            
            # Replicator update
            new_weight = current_weight * np.exp(self.eta * quality)
            new_weights[strategy] = new_weight
            denominator += new_weight
        
        # Normalize weights (ensure they sum to 1)
        if denominator > 1e-10:
            for strategy in self.available_models:
                self.strategy_weights[strategy] = new_weights[strategy] / denominator
                # Apply epsilon-floor again after normalization
                self.strategy_weights[strategy] = max(self.strategy_weights[strategy], self.epsilon_floor)
        else:
            # Fallback: equal weights if denominator is too small
            equal_weight = 1.0 / len(self.available_models)
            for strategy in self.available_models:
                self.strategy_weights[strategy] = equal_weight
        
        # Renormalize after applying floor
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 1e-10:
            for strategy in self.available_models:
                self.strategy_weights[strategy] /= total_weight
    
    def _apply_strategy(self, strategy):
        """
        Apply the selected strategy by updating pricing_model and pricer.
        """
        if strategy == self.pricing_model:
            return  # Already using this strategy
        
        old_model = self.pricing_model
        self.pricing_model = strategy
        
        # Reinitialize pricer with appropriate parameters
        if strategy == PricingModel.TFBS:
            self.pricer = get_pricer(strategy, alpha=0.85, n_mc=1000)
        elif strategy == PricingModel.HESTON:
            self.pricer = get_pricer(strategy, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, n_mc=1000)
        else:
            self.pricer = get_pricer(strategy)
        
        # Debug: print switch (optional, can be disabled)
        if hasattr(self.model, 'debug') and self.model.debug:
            print(f"Dealer {self.unique_id}: {old_model.value} -> {strategy.value} "
                  f"(step: {self.model.market.book.t})")
    
    def step(self):
        """
        Main step function for option dealer with replicator switching.
        Implements the algorithm from PDF:
        1. Select strategy probabilistically based on weights
        2. Apply strategy (update pricing_model)
        3. Quote and hedge using selected strategy
        4. Calculate reward
        5. Update Q_s for selected strategy
        6. Update weights using replicator dynamics
        """
        # Step 1: Select strategy probabilistically 
        if self.enable_model_switching:
            selected_strategy = self._select_strategy()
            strategy_switched = (selected_strategy != self.current_strategy)
            self._apply_strategy(selected_strategy)
            self.current_strategy = selected_strategy
            self._strategy_switched_this_step = strategy_switched
        else:
            selected_strategy = self.pricing_model
            self._strategy_switched_this_step = False
        
        # Step 2: Check if we received any option trades (our quotes were hit)
        self._check_option_trades()
        
        # Step 3: Update quotes using current strategy
        self._update_quotes()
        
        # Step 4: Hedge periodically (or every step if frequency is 1)
        if (self.model.market.book.t - self.last_hedge_step) >= self.hedge_frequency:
            self._hedge_position()
        
        # Step 5: Calculate reward for this step
        reward = self._calculate_reward()
        self.current_reward = reward  # Store for logging
        
        # Step 6: Update quality Q_s for selected strategy (exponential smoothing)
        if self.enable_model_switching:
            self._update_strategy_quality(selected_strategy, reward)
        
        # Step 7: Update weights using replicator dynamics
        if self.enable_model_switching:
            self._update_strategy_weights()
        
        # Note: Don't reset _strategy_switched_this_step here - it's used for logging
        # It will be set again at the start of next step


class OptionTaker(Agent):
    """Agent that occasionally trades options (hits the book)."""
    
    def __init__(self, uid, model, contract_ids, trade_prob=0.01, max_qty=5):
        super().__init__(model)
        self.contract_ids = list(contract_ids)
        self.trade_prob = float(trade_prob)
        self.max_qty = int(max_qty)
    
    def step(self):
        """Occasionally place market orders in options market."""
        if not hasattr(self.model, 'options_market'):
            return
        
        if float(self.model.rng.uniform()) > self.trade_prob:
            return
        
        # Randomly select a contract
        contract_id = self.model.rng.choice(self.contract_ids)
        contract = self.model.options_market.get_contract(contract_id)
        if contract is None:
            return
        
        # Random side (slightly biased toward puts in stress)
        reg = int(self.model.regime)
        put_bias = 0.6 if reg == 1 else 0.5
        
        if contract.option_type == "put" and float(self.model.rng.uniform()) < put_bias:
            side = "buy"  # Buy puts (protective)
        else:
            side = "buy" if float(self.model.rng.uniform()) < 0.5 else "sell"
        
        qty = int(self.model.rng.integers(1, self.max_qty + 1))
        self.model.options_market.place_market(contract_id, self.unique_id, side, qty)

