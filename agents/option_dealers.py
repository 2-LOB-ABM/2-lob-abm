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
        r=0.0,
        base_spread_pct=0.02,
        quote_size=10,
        hedge_frequency=1,  # Hedge every N steps (1 = every step)
        inventory_risk_aversion=0.1,
        enable_model_switching=True,
        switching_window=50,  # Steps to evaluate performance
        switching_threshold=0.05,  # Minimum improvement to switch
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
        
        # Initialize pricer
        self.pricer = get_pricer(pricing_model, **pricer_kwargs)
        
        # Inventory tracking: contract_id -> (position, avg_price)
        self.inventory = defaultdict(lambda: {"position": 0, "avg_price": 0.0})
        
        # Hedging: delta position in underlying
        self.hedge_position = 0.0  # Net delta hedge position
        
        # Performance tracking for model switching
        self.hedge_errors = []  # List of (step, error) tuples
        self.pnl_history = []  # Cumulative PnL history
        self.total_pnl = 0.0  # Total PnL
        self.last_hedge_step = 0
        self.last_hedge_price = None  # Price when last hedge was executed
        self.last_option_value = 0.0  # Last calculated option portfolio value
        self.last_delta = 0.0  # Last delta position
        
        # Track when we get new option positions (trades executed against our quotes)
        self.trades_received = []  # List of (step, contract_id, side, qty, price)
        
        # Inertia: prevent switching back too quickly
        self.last_switch_step = -1000  # Step when we last switched models
        self.switch_cooldown = 20  # Minimum steps between switches
        
        # Track error trend for trend-based switching
        self.error_trend = []  # List of recent errors to detect trends
        self.trend_window = 5  # Number of steps to check for increasing trend
        
        # Live quotes tracking
        self.live_quotes = defaultdict(list)  # contract_id -> [order_ids]
        
        # Model switching candidates
        self.available_models = [PricingModel.BS, PricingModel.TFBS, PricingModel.HESTON]
        # Track hedging errors for ALL models (not just current one)
        # This allows us to compare models and choose the best one
        self.model_hedge_errors = {m: [] for m in self.available_models}
        # Store pricers for all models to evaluate them in parallel
        self.model_pricers = {}
        for model_type in self.available_models:
            if model_type == PricingModel.TFBS:
                self.model_pricers[model_type] = get_pricer(model_type, alpha=0.85, n_mc=5000)
            elif model_type == PricingModel.HESTON:
                self.model_pricers[model_type] = get_pricer(model_type, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, n_mc=5000)
            else:
                self.model_pricers[model_type] = get_pricer(model_type)
        
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
        
        # Adjust for inventory risk
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
        Calculate theoretical hedging error for a given model.
        Compares how well this model would have hedged our actual positions.
        Uses the same positions for all models to ensure fair comparison.
        """
        if model_type not in self.model_pricers:
            return 0.0
        
        if abs(price_change) < 1e-6:
            return 0.0
        
        S = float(self.model.current_price)
        sigma = float(max(self.model.volatility, 1e-6))
        pricer = self.model_pricers[model_type]
        total_error = 0.0
        
        # Calculate error for each contract in our ACTUAL inventory
        # This ensures we compare models fairly using the same positions
        for contract_id, inv_data in self.inventory.items():
            contract = self.model.options_market.get_contract(contract_id)
            if contract is None:
                continue
            
            position = inv_data["position"]
            if abs(position) < 0.01:
                continue
            
            # Calculate current delta for this model
            if model_type == PricingModel.BS:
                if contract.option_type == "call":
                    delta_now = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity)
                else:
                    delta_now = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity)
            elif model_type == PricingModel.TFBS:
                seed = self.model.rng.integers(0, 2**31)
                if contract.option_type == "call":
                    delta_now = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                else:
                    delta_now = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
            elif model_type == PricingModel.HESTON:
                seed = self.model.rng.integers(0, 2**31)
                if contract.option_type == "call":
                    delta_now = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                else:
                    delta_now = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
            else:
                delta_now = 0.0
            
            # Calculate what delta was at last hedge (approximate using current delta)
            # The error is based on how much delta changed vs price change
            # If we had hedged with this model's delta, what would be the error?
            # Error = |actual_option_pnl - hedge_pnl|
            # We approximate option_pnl using delta approximation: position * delta * price_change
            # But we need to compare with what the hedge would have been
            
            # If we had used this model's delta at last hedge, hedge PnL would be:
            # (position * delta_at_last_hedge * price_change)
            # But we don't know delta_at_last_hedge, so we approximate:
            # The error is the difference between how much delta changed
            # For a fair comparison, we use the fact that better models have more stable deltas
            # Error scales with gamma (rate of change of delta)
            
            # Calculate gamma to estimate delta change
            if model_type == PricingModel.BS:
                if contract.option_type == "call":
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity)
                else:
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity)
            elif model_type == PricingModel.TFBS:
                seed = self.model.rng.integers(0, 2**31)
                if contract.option_type == "call":
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                else:
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
            elif model_type == PricingModel.HESTON:
                seed = self.model.rng.integers(0, 2**31)
                if contract.option_type == "call":
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                else:
                    gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
            else:
                gamma = 0.0
            
            # Error is proportional to gamma * price_change^2 (delta changes with price)
            # Models with higher gamma have larger hedging errors when price moves
            delta_change_error = abs(position * gamma * price_change * price_change) * 0.5
            total_error += delta_change_error
        
        # If no inventory, calculate theoretical error based on potential positions
        if total_error == 0.0:
            for contract_id in self.contract_ids:
                contract = self.model.options_market.get_contract(contract_id)
                if contract is None:
                    continue
                
                # Calculate theoretical delta and gamma
                if model_type == PricingModel.BS:
                    if contract.option_type == "call":
                        delta = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity)
                        gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity)
                    else:
                        delta = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity)
                        gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity)
                elif model_type == PricingModel.TFBS:
                    seed = self.model.rng.integers(0, 2**31)
                    if contract.option_type == "call":
                        delta = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                        gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    else:
                        delta = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                        gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                elif model_type == PricingModel.HESTON:
                    seed = self.model.rng.integers(0, 2**31)
                    if contract.option_type == "call":
                        delta = pricer.delta_call(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                        gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                    else:
                        delta = pricer.delta_put(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                        gamma = pricer.gamma(S, contract.strike, self.r, sigma, contract.maturity, seed=seed)
                else:
                    delta = 0.0
                    gamma = 0.0
                
                # Theoretical error scales with gamma (delta sensitivity) and price change
                theoretical_position = 10.0  # Assume 10 contracts
                theoretical_error = abs(theoretical_position * gamma * price_change * price_change) * 0.5
                if abs(price_change) > 0.001:
                    theoretical_error += 0.001
                total_error += theoretical_error
        
        return total_error
    
    def _check_option_trades(self):
        """Check if any of our option quotes were executed by monitoring LOB trades."""
        if not hasattr(self.model, 'options_market'):
            return
        
        current_step = self.model.market.book.t
        
        # Track previous inventory state
        prev_inventory = dict(self.inventory)
        
        # Check each contract's LOB for trades involving our orders
        for contract_id in self.contract_ids:
            lob = self.model.options_market.get_lob(contract_id)
            if lob is None:
                continue
            
            # Check recent trades in this contract's LOB
            for trade in lob.trades:
                step, price, qty, passive_id, aggressive_id = trade
                
                # If we were the passive side (our quote was hit) and this is a recent trade
                if passive_id == self.unique_id and step >= current_step - 1:
                    # Determine side: if our ask was hit, we sold; if our bid was hit, we bought
                    # Check our live quotes to see which side was hit
                    side = None
                    for oid in self.live_quotes.get(contract_id, []):
                        order = lob.orders.get(oid)
                        if order is None:
                            continue
                        # Check if this order was at the trade price
                        order_price = order.price_ticks * lob.tick_size if order.price_ticks else None
                        if order_price and abs(order_price - price) < 0.001:
                            # If order was on ask side (sell), we sold; if bid side (buy), we bought
                            side = order.side
                            break
                    
                    if side:
                        # Record the trade
                        self._record_trade(contract_id, side, qty, price)
                        self.trades_received.append((step, contract_id, side, qty, price))
    
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
        
        # Track hedging error for CURRENT model and ALL other models
        # This allows us to compare models and choose the best one
        if self.last_hedge_price is not None and self.last_hedge_step > 0:
            price_change = current_price - self.last_hedge_price
            
            # Calculate error for current model
            if abs(self.last_option_value) > 0.01 or abs(current_option_value) > 0.01:
                # We have inventory - track actual hedging error
                hedge_pnl = self.hedge_position * price_change
                option_pnl = current_option_value - self.last_option_value
                hedge_error = option_pnl - hedge_pnl
                delta_error = abs(total_delta - self.last_delta) * abs(price_change) * 0.1
                total_error = abs(hedge_error) + delta_error
                
                self.hedge_errors.append((self.model.market.book.t, total_error))
                self.pnl_history.append(option_pnl - hedge_pnl)
                self.total_pnl += (option_pnl - hedge_pnl)
                
                # Track error trend
                self.error_trend.append(total_error)
                if len(self.error_trend) > self.trend_window:
                    self.error_trend.pop(0)  # Keep only recent errors
            else:
                # No inventory - calculate theoretical error
                theoretical_error = self._calculate_model_error(self.pricing_model, price_change)
                if theoretical_error > 0:
                    self.hedge_errors.append((self.model.market.book.t, theoretical_error))
                    if len(self.hedge_errors) > self.switching_window * 2:
                        self.hedge_errors = self.hedge_errors[-self.switching_window:]
                    self.error_trend.append(theoretical_error)
                    if len(self.error_trend) > self.trend_window:
                        self.error_trend.pop(0)
            
            # Calculate errors for ALL models to compare performance
            for model_type in self.available_models:
                model_error = self._calculate_model_error(model_type, price_change)
                if model_error > 0:
                    self.model_hedge_errors[model_type].append((self.model.market.book.t, model_error))
                    # Keep error lists manageable
                    if len(self.model_hedge_errors[model_type]) > self.switching_window * 2:
                        self.model_hedge_errors[model_type] = self.model_hedge_errors[model_type][-self.switching_window:]
        
        # Execute hedge
        if abs(delta_to_hedge) > 0.01:  # Threshold to avoid micro-trades
            # Execute hedge as market order
            qty = int(abs(delta_to_hedge))
            side = "buy" if delta_to_hedge > 0 else "sell"
            self.model.market.place_market(self.unique_id, side, qty)
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
    
    def _evaluate_model_performance(self):
        """
        Evaluate current model performance based on accumulated hedging errors.
        Implements evolutionary switching: dealers with poor performance switch models.
        """
        if not self.enable_model_switching:
            return
        
        # Check cooldown - don't switch too frequently
        steps_since_switch = self.model.market.book.t - self.last_switch_step
        if steps_since_switch < self.switch_cooldown:
            return
        
        # Need some data to evaluate, but allow evaluation with less data too
        min_errors_needed = max(5, self.switching_window // 10)  # Reduced: at least 5 errors or 10% of window
        if len(self.hedge_errors) < min_errors_needed:
            # Still allow regime-based switching even with few errors
            pass
        else:
            # Use available errors (may be less than full window)
            errors_to_use = min(len(self.hedge_errors), self.switching_window)
        
        # Calculate performance metrics using available errors
        if len(self.hedge_errors) >= min_errors_needed:
            recent_errors = [e[1] for e in self.hedge_errors[-errors_to_use:]]
            mean_error = np.mean(recent_errors)
            max_error = np.max(recent_errors) if recent_errors else 0.0
            
            # Also consider total PnL as a performance indicator
            recent_pnl = self.pnl_history[-errors_to_use:] if len(self.pnl_history) >= errors_to_use else self.pnl_history
            mean_pnl = np.mean(recent_pnl) if recent_pnl else 0.0
        else:
            # Not enough errors yet - use defaults
            mean_error = 0.0
            max_error = 0.0
            mean_pnl = 0.0
        
        # Use absolute error values, not normalized (normalization can hide small but significant errors)
        # Typical option price is around 1-10, so errors of 0.01-0.1 are significant
        should_switch = False
        switch_reason = None
        
        # Check for increasing error trend (errors increasing several steps in a row)
        trend_increasing = False
        trend_strength = 0.0
        if len(self.error_trend) >= self.trend_window:
            # Check if errors are consistently increasing
            increasing_count = 0
            for i in range(1, len(self.error_trend)):
                if self.error_trend[i] > self.error_trend[i-1]:
                    increasing_count += 1
            
            # If errors increased in most recent steps, it's a trend
            if increasing_count >= self.trend_window - 1:  # All or almost all steps increasing
                trend_increasing = True
                # Calculate trend strength (relative increase)
                if abs(self.error_trend[0]) > 1e-6:
                    trend_strength = (self.error_trend[-1] - self.error_trend[0]) / abs(self.error_trend[0])
                else:
                    trend_strength = abs(self.error_trend[-1] - self.error_trend[0])
        
        # DETERMINISTIC switching based on error thresholds AND trends
        # Lower thresholds to make switching more sensitive to errors
        if mean_error > 0.03:  # Large error - definitely switch (lowered from 0.05)
            should_switch = True
            switch_reason = "large_error"
        elif mean_error > 0.01:  # Medium error - high probability switch (lowered from 0.02)
            switch_prob = 0.8  # 80% chance
            should_switch = float(self.model.rng.uniform()) < switch_prob
            switch_reason = "medium_error"
        elif mean_error > 0.005 and mean_pnl < -0.005:  # Small error + negative PnL (lowered thresholds)
            switch_prob = 0.6  # 60% chance (increased from 50%)
            should_switch = float(self.model.rng.uniform()) < switch_prob
            switch_reason = "error_and_loss"
        elif trend_increasing:  # Errors increasing consistently over several steps
            # Switch based on trend strength - more sensitive
            if trend_strength > 0.3:  # Strong increasing trend (>30% increase, lowered from 50%)
                should_switch = True
                switch_reason = "strong_increasing_trend"
            elif trend_strength > 0.15:  # Moderate increasing trend (>15% increase, lowered from 20%)
                switch_prob = 0.8  # 80% chance (increased from 70%)
                should_switch = float(self.model.rng.uniform()) < switch_prob
                switch_reason = "moderate_increasing_trend"
            elif trend_strength > 0.05:  # Weak but consistent increasing trend (>5% increase, lowered from 10%)
                switch_prob = 0.5  # 50% chance (increased from 40%)
                should_switch = float(self.model.rng.uniform()) < switch_prob
                switch_reason = "weak_increasing_trend"
        
        # Choose model based ONLY on hedging errors - no regime influence
        # Compare all models and choose the one with lowest average error
        if should_switch:
            other_models = [m for m in self.available_models if m != self.pricing_model]
            
            if not other_models:
                return
            
            # Calculate average error for each model
            model_avg_errors = {}
            for model_type in self.available_models:
                errors = self.model_hedge_errors[model_type]
                if len(errors) >= min_errors_needed:
                    recent_errors = [e[1] for e in errors[-self.switching_window:]]
                    model_avg_errors[model_type] = np.mean(recent_errors)
                else:
                    # Not enough data - use a high default error
                    model_avg_errors[model_type] = float('inf')
            
            # Choose model with lowest average error
            # If multiple models have same error, prefer current model (inertia)
            best_error = min(model_avg_errors.values())
            best_models = [m for m, err in model_avg_errors.items() if err == best_error]
            
            # If current model is best, don't switch
            if self.pricing_model in best_models and len(best_models) == 1:
                return
            
            # Choose from best models (excluding current if it's not the only best)
            candidate_models = [m for m in best_models if m != self.pricing_model]
            if not candidate_models:
                candidate_models = other_models
            
            # If we have error data, choose the best model
            # Otherwise, choose randomly (shouldn't happen if should_switch is True)
            if candidate_models:
                new_model = candidate_models[0] if len(candidate_models) == 1 else self.model.rng.choice(candidate_models)
            else:
                new_model = self.model.rng.choice(other_models)
            
            if new_model != self.pricing_model:
                # Switch model
                old_model = self.pricing_model
                self.pricing_model = new_model
                self.last_switch_step = self.model.market.book.t  # Record switch time
                
                # Reinitialize pricer with appropriate parameters
                if new_model == PricingModel.TFBS:
                    self.pricer = get_pricer(new_model, alpha=0.85, n_mc=5000)
                elif new_model == PricingModel.HESTON:
                    self.pricer = get_pricer(new_model, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, n_mc=5000)
                else:
                    self.pricer = get_pricer(new_model)
                
                # Reset performance tracking (keep some history for continuity)
                # Keep last 25% of history
                keep_size = max(5, self.switching_window // 4)
                self.hedge_errors = self.hedge_errors[-keep_size:]
                self.pnl_history = self.pnl_history[-keep_size:] if len(self.pnl_history) > keep_size else self.pnl_history
                
                # Reset error trend tracking
                self.error_trend = []
                
                # Reset some tracking variables
                self.last_hedge_price = None  # Will recalculate on next hedge
                self.last_option_value = 0.0
                
                # Reset error tracking for new model (keep some history)
                keep_size = max(5, self.switching_window // 4)
                self.model_hedge_errors[new_model] = self.model_hedge_errors[new_model][-keep_size:]
                
                # Debug: print switch (optional, can be disabled)
                if hasattr(self.model, 'debug') and self.model.debug:
                    current_error = model_avg_errors.get(self.pricing_model, 0.0)
                    new_error = model_avg_errors.get(new_model, 0.0)
                    print(f"Dealer {self.unique_id}: {old_model.value} -> {new_model.value} "
                          f"(reason: {switch_reason}, step: {self.model.market.book.t}, "
                          f"current_error: {current_error:.4f}, new_error: {new_error:.4f})")
    
    def step(self):
        """Main step function for option dealer."""
        # Check if we received any option trades (our quotes were hit)
        self._check_option_trades()
        
        # Update quotes
        self._update_quotes()
        
        # Hedge periodically (or every step if frequency is 1)
        # Always hedge to accumulate errors for model evaluation
        if (self.model.market.book.t - self.last_hedge_step) >= self.hedge_frequency:
            self._hedge_position()
        # Also track errors even when not hedging (for continuous evaluation)
        elif self.last_hedge_price is not None:
            # Update option values and track theoretical errors
            current_price = float(self.model.current_price)
            if abs(current_price - self.last_hedge_price) > 0.001:  # Price changed
                # Quick error tracking without full hedge
                current_option_value = 0.0
                for contract_id, inv_data in self.inventory.items():
                    contract = self.model.options_market.get_contract(contract_id)
                    if contract is None:
                        continue
                    res_price = self._get_reservation_price(
                        contract_id, contract.option_type, contract.strike, contract.maturity
                    )
                    current_option_value += inv_data["position"] * res_price
                
                if abs(current_option_value - self.last_option_value) > 0.001:
                    price_change = current_price - self.last_hedge_price
                    option_pnl = current_option_value - self.last_option_value
                    hedge_pnl = self.hedge_position * price_change
                    error = abs(option_pnl - hedge_pnl)
                    if error > 0:
                        self.hedge_errors.append((self.model.market.book.t, error))
                        # Keep list manageable
                        if len(self.hedge_errors) > self.switching_window * 2:
                            self.hedge_errors = self.hedge_errors[-self.switching_window:]
                        
                        # Track error trend
                        self.error_trend.append(error)
                        if len(self.error_trend) > self.trend_window:
                            self.error_trend.pop(0)
        
        # Evaluate model performance and switch if needed (H1)
        # Do this after hedging to accumulate errors
        if self.enable_model_switching:
            self._evaluate_model_performance()


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

