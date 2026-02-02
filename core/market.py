"""
Market wrapper for Limit Order Book with price tracking and statistics.
"""
import numpy as np
from core.lob import LimitOrderBook


class LOBMarket:
    """Market wrapper around LimitOrderBook with price tracking."""
    
    def __init__(self, S0, dt, seed, tick_size=0.01, base_spread_ticks=2):
        self.dt = float(dt)
        self.rng = np.random.default_rng(int(seed))
        self.book = LimitOrderBook(tick_size=tick_size, rng=self.rng)

        self.mid = float(S0)
        self.prices = [float(S0)]
        self.log_returns = []

        self.base_spread_ticks = int(base_spread_ticks)

    def best_bid(self):
        return self.book.best_bid()

    def best_ask(self):
        return self.book.best_ask()

    def spread(self):
        return self.book.spread()

    def place_limit(self, agent_id, side, price, qty, ttl=1):
        return self.book.add_limit(agent_id=agent_id, side=side, price=price, qty=qty, ttl=ttl)

    def place_market(self, agent_id, side, qty):
        return self.book.add_market(agent_id=agent_id, side=side, qty=qty)

    def cancel(self, order_id):
        return self.book.cancel(order_id)

    def end_step(self):
        if self.book.last_trade_price is not None:
            new_price = float(self.book.last_trade_price)
        else:
            new_price = float(self.book.mid_price(self.mid))

        new_price = float(max(new_price, 1e-8))
        r = float(np.log(new_price / self.mid))

        self.mid = new_price
        self.prices.append(self.mid)
        self.log_returns.append(r)

        self.book.last_trade_price = None
        self.book.last_trade_ticks = None

        self.book.step_time()

    def realized_sigma(self, window=50):
        lr = np.array(self.log_returns[-int(window):], dtype=float)
        if lr.size < 2:
            return 0.0
        return float(np.std(lr) / np.sqrt(self.dt))

    def recent_momentum(self, window):
        w = int(window)
        if len(self.log_returns) < w:
            return 0.0
        return float(np.mean(self.log_returns[-w:]))

    def recent_vol(self, window):
        w = int(window)
        if len(self.log_returns) < w:
            return 0.0
        return float(np.std(self.log_returns[-w:]) / np.sqrt(self.dt))
    
    def mark_to_market(self):
        if self.book.last_trade_price is not None:
            self.mid = float(self.book.last_trade_price)
        else:
            self.mid = float(self.book.mid_price(self.mid))

