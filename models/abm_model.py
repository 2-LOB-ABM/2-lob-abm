"""
Unified ABM model with underlying LOB market and options market.
Integrates option dealers with hedging feedback.
"""
import numpy as np
from collections import defaultdict
from mesa import Model

from core.market import LOBMarket
from options.market import OptionsMarket, OptionContract
from options.pricing import PricingModel
from agents.base_agents import Fundamentalist, Chartist, MarketMaker, NoiseTrader, pareto_int
from agents.option_dealers import OptionDealer, OptionTaker


class UnifiedABMModel(Model):
    """
    Unified Agent-Based Model with:
    - Underlying LOB market
    - Options market with dealers
    - Hedging feedback between markets
    - Model switching mechanism (H1)
    """
    
    def __init__(
        self,
        seed,
        S0,
        dt,
        steps,
        # Underlying market agents
        n_fund=10,
        n_chart=10,
        n_mm=3,
        n_noise=50,
        fundamental_price=None,
        # Agent parameters
        fund_strength=0.5,
        chart_strength=0.6,
        chart_vol_sens=2.0,
        mom_window=20,
        vol_window=20,
        agent_noise=0.15,
        # Market parameters
        tick_size=0.1,  # Price step: 100.0 -> 100.1 instead of 100.00 -> 100.01
        base_spread_ticks=2,
        mm_size=15,
        # Options market parameters
        enable_options=True,
        n_option_contracts=3,  # ATM, OTM put
        option_tick_size=0.01,
        n_option_dealers=5,
        dealer_model_distribution=None,  # Dict: {PricingModel: fraction}
        n_option_takers=10,
        option_taker_trade_prob=0.01,
        enable_model_switching=True,
        # Regime switching parameters
        p01=0.02,
        p10=0.10,
        shock_rate=0.01,
        shock_impact=8.0,
        n_events_calm=400,
        n_events_stress=1200,
        xi_wr=1.0,
        xi_ws=1.0,
        xi_wI=1.0,
        theta_r=3.0,
        theta_s=None,
        theta_I=0.5,
        # MM reaction parameters
        mm_react_prob=0.6,
        mm_latency_events=2,
        # Hawkes process parameters
        hawkes_mu=200.0,
        hawkes_alpha=0.5,
        hawkes_beta=5.0,
        max_events=3000,
        # Debug
        debug=False,
        debug_print_every=0,
        debug_snapshot_every=0,
        debug_l2_depth=10,
        **cfg
    ):
        super().__init__()
        self.rng = np.random.default_rng(int(seed))
        self.steps_n = int(steps)
        self.dt = float(dt)
        self.tick_size = float(tick_size)
        
        # Initialize underlying market
        self.market = LOBMarket(
            S0=float(S0),
            dt=float(dt),
            seed=int(seed) + 1,
            tick_size=float(tick_size),
            base_spread_ticks=int(base_spread_ticks)
        )
        
        self.current_price = float(S0)
        self.fundamental_price = float(fundamental_price if fundamental_price is not None else S0)
        self.volatility = 0.2  # Initial volatility estimate (20% annualized)
        
        # Regime switching
        self.regime = 0
        self.p01 = float(p01)
        self.p10 = float(p10)
        self.shock_rate = float(shock_rate)
        self.shock_impact = float(shock_impact)
        self.n_events_calm = int(n_events_calm)
        self.n_events_stress = int(n_events_stress)
        self.xi_wr = float(xi_wr)
        self.xi_ws = float(xi_ws)
        self.xi_wI = float(xi_wI)
        self.theta_r = float(theta_r)
        self.theta_s = float(theta_s if theta_s is not None else 3.0 * self.tick_size)
        self.theta_I = float(theta_I)
        
        # MM reaction
        self.mm_react_prob = float(mm_react_prob)
        self.mm_latency_events = int(mm_latency_events)
        self._mm_pending = 0
        self.mm_react_log = []
        
        # Metaorder parameters
        self.meta_left = 0
        self.meta_side = "sell"
        self.meta_intensity = 0
        
        # Hawkes process
        self.hawkes_mu = float(hawkes_mu)
        self.hawkes_alpha = float(hawkes_alpha)
        self.hawkes_beta = float(hawkes_beta)
        self.hawkes_H = 0.0
        self.max_events = int(max_events)
        
        # Options market
        self.enable_options = bool(enable_options)
        self.options_market = None
        self.option_contracts = []
        
        if self.enable_options:
            self._initialize_options_market(
                n_contracts=int(n_option_contracts),
                option_tick_size=float(option_tick_size),
                S0=float(S0)
            )
        
        # Agents
        self.agents_list = []
        self._initialize_agents(
            n_fund=int(n_fund),
            n_chart=int(n_chart),
            n_mm=int(n_mm),
            n_noise=int(n_noise),
            fund_strength=float(fund_strength),
            chart_strength=float(chart_strength),
            chart_vol_sens=float(chart_vol_sens),
            mom_window=int(mom_window),
            vol_window=int(vol_window),
            agent_noise=float(agent_noise),
            base_spread_ticks=int(base_spread_ticks),
            mm_size=int(mm_size),
            n_option_dealers=int(n_option_dealers),
            dealer_model_distribution=dealer_model_distribution,
            n_option_takers=int(n_option_takers),
            option_taker_trade_prob=float(cfg.get("option_taker_trade_prob", cfg.get("trade_prob", 0.01))),
            enable_model_switching=bool(enable_model_switching)
        )
        
        # Debug and logging
        self.debug = bool(debug)
        self.debug_print_every = int(debug_print_every)
        self.debug_snapshot_every = int(debug_snapshot_every)
        self.debug_l2_depth = int(debug_l2_depth)
        
        self._initialize_logging()
    
    def _initialize_options_market(self, n_contracts, option_tick_size, S0):
        """Initialize options market with contracts."""
        self.options_market = OptionsMarket(
            tick_size=option_tick_size,
            rng=self.rng
        )
        
        # Create contracts: ATM call, ATM put, OTM put
        contract_id = 1
        T = float(self.steps_n) * self.dt
        
        # ATM call
        if n_contracts >= 1:
            contract = OptionContract(
                contract_id=contract_id,
                option_type="call",
                strike=float(S0),
                maturity=T,
                underlying_price=float(S0)
            )
            self.options_market.add_contract(contract)
            self.option_contracts.append(contract)
            contract_id += 1
        
        # ATM put
        if n_contracts >= 2:
            contract = OptionContract(
                contract_id=contract_id,
                option_type="put",
                strike=float(S0),
                maturity=T,
                underlying_price=float(S0)
            )
            self.options_market.add_contract(contract)
            self.option_contracts.append(contract)
            contract_id += 1
        
        # OTM put (strike = 0.9 * S0)
        if n_contracts >= 3:
            contract = OptionContract(
                contract_id=contract_id,
                option_type="put",
                strike=float(S0 * 0.9),
                maturity=T,
                underlying_price=float(S0)
            )
            self.options_market.add_contract(contract)
            self.option_contracts.append(contract)
    
    def _initialize_agents(
        self,
        n_fund, n_chart, n_mm, n_noise,
        fund_strength, chart_strength, chart_vol_sens,
        mom_window, vol_window, agent_noise,
        base_spread_ticks, mm_size,
        n_option_dealers, dealer_model_distribution, n_option_takers, enable_model_switching, option_taker_trade_prob
    ):
        """Initialize all agents."""
        uid = 0
        
        # Underlying market makers
        for _ in range(n_mm):
            self.agents_list.append(
                MarketMaker(uid, self, base_spread_ticks=base_spread_ticks, size=mm_size, ttl=5, ttl_jitter=5)
            )
            uid += 1
        
        # Fundamentalists
        for _ in range(n_fund):
            self.agents_list.append(
                Fundamentalist(uid, self, strength=fund_strength, noise_scale=agent_noise)
            )
            uid += 1
        
        # Chartists
        for _ in range(n_chart):
            self.agents_list.append(
                Chartist(uid, self, strength=chart_strength, vol_sens=chart_vol_sens,
                        mom_window=mom_window, vol_window=vol_window, noise_scale=agent_noise)
            )
            uid += 1
        
        # Noise traders
        for _ in range(n_noise):
            self.agents_list.append(
                NoiseTrader(uid, self, prob=0.5, sell_bias=0.7)
            )
            uid += 1
        
        # Option dealers
        if self.enable_options and len(self.option_contracts) > 0:
            contract_ids = [c.contract_id for c in self.option_contracts]
            
            # Distribute models among dealers
            if dealer_model_distribution is None:
                # Default: equal distribution
                models = [PricingModel.BS, PricingModel.TFBS, PricingModel.HESTON]
                dealer_model_distribution = {m: 1.0 / len(models) for m in models}
            
            # Use random assignment instead of deterministic distribution
            # This ensures variability even with same initial distribution
            model_list = []
            for i in range(n_option_dealers):
                # Randomly assign model based on distribution
                rand_val = float(self.rng.uniform())
                cumsum = 0.0
                assigned_model = PricingModel.BS  # Default
                for model_type, fraction in dealer_model_distribution.items():
                    cumsum += fraction
                    if rand_val <= cumsum:
                        assigned_model = model_type
                        break
                model_list.append(assigned_model)
            
            # Shuffle for additional randomness
            self.rng.shuffle(model_list)
            
            for i in range(n_option_dealers):
                model_type = model_list[i] if i < len(model_list) else PricingModel.BS
                
                # Model-specific parameters with some randomness for variability
                pricer_kwargs = {}
                if model_type == PricingModel.TFBS:
                    # Add small random variation to alpha
                    alpha = 0.85 + self.rng.uniform(-0.05, 0.05)
                    alpha = max(0.7, min(1.0, alpha))  # Keep in valid range
                    pricer_kwargs = {"alpha": alpha, "n_mc": 5000}
                elif model_type == PricingModel.HESTON:
                    # Add small random variation to Heston parameters
                    kappa = 2.0 + self.rng.uniform(-0.3, 0.3)
                    theta = 0.04 + self.rng.uniform(-0.01, 0.01)
                    sigma_v = 0.3 + self.rng.uniform(-0.05, 0.05)
                    rho = -0.7 + self.rng.uniform(-0.1, 0.1)
                    pricer_kwargs = {
                        "kappa": max(0.5, kappa),
                        "theta": max(0.01, theta),
                        "sigma_v": max(0.1, sigma_v),
                        "rho": max(-0.9, min(-0.3, rho)),
                        "n_mc": 5000
                    }
                
                dealer = OptionDealer(
                    uid, self, model_type, contract_ids,
                    enable_model_switching=enable_model_switching,
                    **pricer_kwargs
                )
                self.agents_list.append(dealer)
                uid += 1
            
            # Option takers
            for _ in range(n_option_takers):
                self.agents_list.append(
                    OptionTaker(uid, self, contract_ids, trade_prob=option_taker_trade_prob, max_qty=5)
                )
                uid += 1
    
    def _initialize_logging(self):
        """Initialize logging arrays."""
        self.n_events_log = []
        self.mid_micro_log = []
        self.Nn_log = []
        self.lambda_log = []
        self.lambda_reg_log = []
        self.hawkes_H_log = []
        self.hawkes_cap_hit_log = []
        self.bb_none_log = []
        self.ba_none_log = []
        self.order_count_log = []
        self.bid_levels_log = []
        self.ask_levels_log = []
        self.crossed_log = []
        self.n_limit_log = []
        self.n_market_log = []
        self.n_cancel_log = []
        self.n_expire_log = []
        self.meta_active_log = []
        self.meta_left_log = []
        self.meta_intensity_log = []
        self.meta_side_log = []
        self.l2_snapshots = []
        self.regime_log = []
        self.spread_log = []
        self.depth_bid_log = []
        self.depth_ask_log = []
        self.imbalance_log = []
        self.trade_count_log = []
        self.volume_log = []
        self.mm_requotes_log = []
        self.mm_requotes_in_step = 0
        
        # Options market logs
        if self.enable_options:
            self.option_spreads_log = defaultdict(list)
            self.option_depths_log = defaultdict(list)
            self.option_trades_log = defaultdict(list)
            self.dealer_model_distribution_log = []
            # Log hedging errors for all models
            self.dealer_hedge_errors_log = defaultdict(list)  # model -> list of (step, avg_error)
            self.dealer_hedge_errors_by_model = defaultdict(list)  # model -> list of errors over time
    
    def update_regime(self):
        """Update market regime (calm/stress) based on indicators."""
        last_r = self.market.log_returns[-1] if len(self.market.log_returns) else 0.0
        bb = self.market.book.best_bid()
        ba = self.market.book.best_ask()
        
        spread = float(ba - bb) if (bb is not None and ba is not None) else 0.0
        depth_bid, depth_ask = self.market.book.depth_at_best()
        
        imb = (depth_bid - depth_ask) / max(1.0, depth_bid + depth_ask)
        
        # Stress indicator
        r_flag = 1.0 if last_r < -self.theta_r * self.market.realized_sigma(window=50) * self.dt**0.5 else 0.0
        s_flag = 1.0 if spread > self.theta_s else 0.0
        I_flag = 1.0 if abs(imb) > self.theta_I else 0.0
        
        shock_trigger = self.xi_wr * r_flag + self.xi_ws * s_flag + self.xi_wI * I_flag
        
        if self.regime == 0:
            # calm -> stress
            # Reduce sensitivity to shock_trigger to make transitions less frequent
            p = min(1.0, self.p01 * (1.0 + shock_trigger * 0.5))  # Reduced multiplier from 1.0 to 0.5
            if float(self.rng.uniform()) < p:
                self.regime = 1
        else:
            # stress -> calm
            # Make recovery slower - reduce probability when stress indicators are still high
            recovery_factor = 1.0 - min(0.5, shock_trigger * 0.3)  # Slower recovery if stress indicators persist
            p = self.p10 * recovery_factor
            if float(self.rng.uniform()) < p:
                self.regime = 0
    
    def maybe_shock(self):
        """Generate exogenous shocks (metaorders)."""
        if self.meta_left <= 0:
            if float(self.rng.uniform()) >= self.shock_rate:
                return
            
            # Make shocks longer-lasting but less frequent
            # In calm: shorter shocks (40-80 steps), in stress: longer shocks (80-200 steps)
            base_duration = int(self.rng.integers(40, 80))  # Increased from 20-120
            self.meta_left = base_duration * (2 if self.regime == 1 else 1)
            self.meta_side = "sell" if float(self.rng.uniform()) < 0.6 else "buy"
            # Moderate intensity - not too extreme
            self.meta_intensity = int(self.rng.integers(5, 12)) * (2 if self.regime == 1 else 1)  # Reduced multiplier
            self.regime = 1
        
        for _ in range(self.meta_intensity):
            q = int(pareto_int(self.rng, 5.0, 1.2, 20000))
            self.market.place_market(agent_id=-777, side=self.meta_side, qty=q)
        
        self.meta_left -= 1
    
    def step(self):
        """Main simulation step."""
        self.update_regime()
        self.maybe_shock()
        self.mm_requotes_in_step = 0
        
        # Determine number of events (Hawkes process)
        base_scale = self.n_events_calm if self.regime == 0 else self.n_events_stress
        Lambda = self.hawkes_mu + self.hawkes_alpha * self.hawkes_H
        Lambda_reg = Lambda * (base_scale / max(1.0, self.n_events_calm))
        
        Nn = int(self.rng.poisson(Lambda_reg * self.dt))
        n_events = min(self.max_events, max(1, Nn))
        
        # Logging
        self.n_events_log.append(int(n_events))
        self.Nn_log.append(int(Nn))
        self.lambda_log.append(float(Lambda))
        self.lambda_reg_log.append(float(Lambda_reg))
        self.hawkes_H_log.append(float(self.hawkes_H))
        self.hawkes_cap_hit_log.append(int(Nn >= self.max_events))
        
        # Market makers react at beginning
        n_mm_initial = len([a for a in self.agents_list if isinstance(a, MarketMaker)])
        for i in range(n_mm_initial):
            if i < len(self.agents_list):
                self.agents_list[i].step()
        
        # Process micro-events
        mm_reacts_this_step = 0
        for k in range(n_events):
            # Select random agent (excluding initial MMs)
            agent_idx = int(self.rng.integers(n_mm_initial, len(self.agents_list)))
            a = self.agents_list[agent_idx]
            a.step()
            
            trade_now = self.market.book.trade_happened
            if trade_now:
                self.market.mark_to_market()
                self.current_price = float(self.market.mid)
                
                # Micro mid log
                bb = self.market.book.best_bid()
                ba = self.market.book.best_ask()
                t_micro = float(self.market.book.t) + (k + 1) / (n_events + 1)
                
                if bb is not None and ba is not None:
                    mid_q = 0.5 * (float(bb) + float(ba))
                else:
                    mid_q = float(self.market.mid)
                
                self.mid_micro_log.append((t_micro, mid_q))
                
                # MM reaction logic
                if self._mm_pending <= 0 and float(self.rng.uniform()) < self.mm_react_prob:
                    self._mm_pending = self.mm_latency_events
                else:
                    self._mm_pending = max(0, self._mm_pending - 1)
                
                self.market.book.trade_happened = False
            
            # Process pending MM reactions
            if self._mm_pending > 0:
                self._mm_pending -= 1
                if self._mm_pending == 0:
                    for i in range(n_mm_initial):
                        if i < len(self.agents_list):
                            self.agents_list[i].step()
                    mm_reacts_this_step += 1
        
        self.mm_react_log.append(int(mm_reacts_this_step))
        
        # End step
        self.market.end_step()
        
        # Update volatility
        self.current_price = float(self.market.mid)
        self.volatility = float(self.market.realized_sigma(window=50))
        
        # Update options market
        if self.enable_options:
            self.options_market.step_time()
            expired = self.options_market.update_contract_maturities(self.dt)
            if expired:
                # Remove dealers/takers for expired contracts
                pass  # Could implement cleanup here
        
        # Logging
        self._log_step()
        
        # Update Hawkes intensity
        self.hawkes_H = np.exp(-self.hawkes_beta * self.dt) * self.hawkes_H + float(n_events)
    
    def _log_step(self):
        """Log step statistics."""
        bb = self.market.book.best_bid()
        ba = self.market.book.best_ask()
        spread = float(ba - bb) if (bb is not None and ba is not None) else float("nan")
        depth_bid, depth_ask = self.market.book.depth_at_best()
        imb = (depth_bid - depth_ask) / max(1.0, depth_bid + depth_ask)
        
        self.regime_log.append(int(self.regime))
        self.spread_log.append(spread)
        self.depth_bid_log.append(float(depth_bid))
        self.depth_ask_log.append(float(depth_ask))
        self.imbalance_log.append(float(imb))
        self.trade_count_log.append(int(self.market.book.trades_in_step))
        self.volume_log.append(float(self.market.book.volume_in_step))
        self.bb_none_log.append(int(bb is None))
        self.ba_none_log.append(int(ba is None))
        self.order_count_log.append(int(len(self.market.book.orders)))
        self.bid_levels_log.append(int(len(self.market.book.bid_prices)))
        self.ask_levels_log.append(int(len(self.market.book.ask_prices)))
        self.crossed_log.append(int(self.market.book.crossed_in_step > 0))
        self.mm_requotes_log.append(int(self.mm_requotes_in_step))
        self.n_limit_log.append(int(self.market.book.n_limit_in_step))
        self.n_market_log.append(int(self.market.book.n_market_in_step))
        self.n_cancel_log.append(int(self.market.book.n_cancel_in_step))
        self.n_expire_log.append(int(self.market.book.n_expire_in_step))
        self.meta_active_log.append(int(self.meta_left > 0))
        self.meta_left_log.append(int(self.meta_left))
        self.meta_intensity_log.append(int(self.meta_intensity))
        self.meta_side_log.append(1 if self.meta_side == "buy" else -1)
        
        # Options market logging
        if self.enable_options:
            for contract in self.option_contracts:
                cid = contract.contract_id
                spread = self.options_market.get_spread(cid)
                depth = self.options_market.get_depth(cid)
                self.option_spreads_log[cid].append(spread if spread is not None else float("nan"))
                self.option_depths_log[cid].append(depth)
            
            # Log dealer model distribution
            dealers = [a for a in self.agents_list if isinstance(a, OptionDealer)]
            model_counts = {PricingModel.BS: 0, PricingModel.TFBS: 0, PricingModel.HESTON: 0}
            for dealer in dealers:
                model_counts[dealer.pricing_model] = model_counts.get(dealer.pricing_model, 0) + 1
            total = len(dealers)
            if total > 0:
                self.dealer_model_distribution_log.append({
                    m: count / total for m, count in model_counts.items()
                })
            else:
                self.dealer_model_distribution_log.append({})
            
            # Log average hedging errors for each model across all dealers
            # This helps understand why switching happens
            model_errors_sum = {PricingModel.BS: [], PricingModel.TFBS: [], PricingModel.HESTON: []}
            for dealer in dealers:
                if hasattr(dealer, 'model_hedge_errors'):
                    for model_type in [PricingModel.BS, PricingModel.TFBS, PricingModel.HESTON]:
                        errors = dealer.model_hedge_errors.get(model_type, [])
                        if errors:
                            # Get recent errors (last switching_window)
                            recent_errors = [e[1] for e in errors[-dealer.switching_window:]]
                            if recent_errors:
                                model_errors_sum[model_type].extend(recent_errors)
            
            # Calculate average error per model
            for model_type in [PricingModel.BS, PricingModel.TFBS, PricingModel.HESTON]:
                if model_errors_sum[model_type]:
                    avg_error = np.mean(model_errors_sum[model_type])
                    self.dealer_hedge_errors_by_model[model_type].append(avg_error)
                else:
                    self.dealer_hedge_errors_by_model[model_type].append(0.0)
        
        if self.debug_print_every > 0 and (self.market.book.t % self.debug_print_every == 0):
            print(
                "t", self.market.book.t,
                "reg", self.regime,
                "mid", round(self.current_price, 4),
                "spread", round(self.spread_log[-1], 4),
                "vol", round(self.volatility, 4),
                "events", self.n_events_log[-1]
            )
        
        self.market.book.reset_step_counters()
        if self.enable_options:
            self.options_market.reset_step_counters()
    
    def run(self):
        """Run the simulation."""
        for _ in range(self.steps_n):
            self.step()
        return np.array(self.market.prices, dtype=float)

