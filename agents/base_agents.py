"""
Base agents for the underlying market: Fundamentalists, Chartists, Noise Traders, Market Makers.
"""
import numpy as np
from mesa import Agent


def pareto_int(rng, xm, alpha, cap):
    """Sample Pareto-distributed integer."""
    u = float(rng.uniform())
    x = xm / (u ** (1.0 / alpha))
    return int(min(cap, max(1.0, x)))


class Fundamentalist(Agent):
    """Fundamentalist agent that trades based on fundamental price."""
    
    def __init__(self, uid, model, strength, noise_scale, max_qty=5):
        super().__init__(model)
        self.strength = float(strength)
        self.noise_scale = float(noise_scale)
        self.max_qty = int(max_qty)

    def step(self):
        S = float(self.model.current_price)
        F = float(self.model.fundamental_price)
        eps = float(self.model.rng.normal(0.0, 1.0))

        signal = self.strength * (F - S) / max(S, 1e-12) + self.noise_scale * eps
        qty = int(min(self.max_qty, max(1, round(abs(signal) * self.max_qty))))
        side = "buy" if signal > 0 else "sell"
        self.model.market.place_market(self.unique_id, side, qty)


class Chartist(Agent):
    """Momentum/trend-following agent."""
    
    def __init__(self, uid, model, strength, vol_sens, mom_window, vol_window, noise_scale, max_qty=5):
        super().__init__(model)
        self.strength = float(strength)
        self.vol_sens = float(vol_sens)
        self.mom_window = int(mom_window)
        self.vol_window = int(vol_window)
        self.noise_scale = float(noise_scale)
        self.max_qty = int(max_qty)
        self.uid = uid

    def step(self):
        mom = float(self.model.market.recent_momentum(self.mom_window))
        v = float(self.model.market.recent_vol(self.vol_window))
        eps = float(self.model.rng.normal(0.0, 1.0))

        signal = self.strength * mom * (1.0 + self.vol_sens * v) + self.noise_scale * eps
        qty = int(min(self.max_qty, max(1, round(abs(signal) * self.max_qty))))
        side = "buy" if signal > 0 else "sell"
        self.model.market.place_market(self.unique_id, side, qty)


class NoiseTrader(Agent):
    """Noise trader with regime-dependent behavior."""
    
    def __init__(self, uid, model, prob=0.2, sell_bias=0.5, xm=1.0, alpha=1.6, cap=200, limit_offset_ticks=2, ttl=20):
        super().__init__(model)
        self.prob = float(prob)
        self.sell_bias = float(sell_bias)
        self.xm = float(xm)
        self.alpha = float(alpha)
        self.cap = int(cap)
        self.limit_offset_ticks = int(limit_offset_ticks)
        self.ttl = int(ttl)

    def step(self):
        if float(self.model.rng.uniform()) > self.prob:
            return

        reg = int(self.model.regime)
        sell_bias = min(0.95, self.sell_bias + 0.20 * reg)
        side = "sell" if float(self.model.rng.uniform()) < sell_bias else "buy"

        qty = pareto_int(self.model.rng, self.xm, self.alpha, self.cap)

        mid = float(self.model.current_price)
        tick = float(self.model.tick_size)
        off = float(self.limit_offset_ticks + reg) * tick
        limit_prob = 0.85 if reg == 0 else 0.35

        if float(self.model.rng.uniform()) < limit_prob:
            px = mid - off if side == "buy" else mid + off
            self.model.market.place_limit(self.unique_id, side, px, qty, ttl=self.ttl)
        else:
            self.model.market.place_market(self.unique_id, side, qty)


class MarketMaker(Agent):
    """Market maker for the underlying market."""
    
    def __init__(self, uid, model, base_spread_ticks=2, size=10, ttl=10, levels=5, requote_move_ticks=5, ttl_jitter=0):
        super().__init__(model)
        self.base_spread_ticks = int(base_spread_ticks)
        self.size = int(size)
        self.ttl = int(ttl)
        self.live = []
        self.levels = int(levels)
        self.requote_move_ticks = int(requote_move_ticks)
        self.last_mid_ticks = None
        self.ttl_jitter = int(ttl_jitter)

    def step(self):
        book_orders = self.model.market.book.orders
        self.live = [oid for oid in self.live if (oid in book_orders and book_orders[oid].qty > 0)]

        reg = int(self.model.regime)
        mid = self.model.market.book.mid_price(self.model.market.mid)
        tick = float(self.model.tick_size)
        mid_ticks = int(round(mid / tick))
        bb = self.model.market.book.best_bid_ticks()
        ba = self.model.market.book.best_ask_ticks()
        db, da = self.model.market.book.depth_at_best()

        desired_levels = max(1, self.levels - (2 if reg == 1 else 0))
        thr = max(1, self.requote_move_ticks - (1 if reg == 1 else 0))

        need_requote = False
        if self.last_mid_ticks is None:
            need_requote = True
        elif abs(mid_ticks - self.last_mid_ticks) >= thr:
            need_requote = True
        elif bb is None or ba is None:
            need_requote = True
        elif db == 0.0 or da == 0.0:
            need_requote = True
        elif len(self.live) < 2 * desired_levels:
            need_requote = True
        
        if not need_requote:
            return

        self.last_mid_ticks = mid_ticks
        self.model.mm_requotes_in_step += 1

        spread = self.base_spread_ticks + (3 if reg == 1 else 0)
        levels = max(1, self.levels - (2 if reg == 1 else 0))
        base_size = max(1, self.size // (2 if reg == 1 else 1))

        bid0 = mid_ticks - spread
        ask0 = mid_ticks + spread

        book_orders = self.model.market.book.orders

        new_live = []
        buy_ticks = set()
        sell_ticks = set()

        for oid in self.live:
            o = book_orders.get(oid)
            if o is None or o.qty <= 0 or o.price_ticks is None:
                continue

            pt = int(o.price_ticks)
            if o.side == "buy":
                if pt < bid0 - (levels - 1) or pt > bid0:
                    self.model.market.cancel(oid)
                else:
                    buy_ticks.add(pt)
                    new_live.append(oid)
            else:
                if pt < ask0 or pt > ask0 + (levels - 1):
                    self.model.market.cancel(oid)
                else:
                    sell_ticks.add(pt)
                    new_live.append(oid)

        self.live = new_live

        for L in range(levels):
            q = max(1, base_size // (L + 1))

            bid_ticks_L = bid0 - L
            ask_ticks_L = ask0 + L
            ttl = self.ttl + int(self.model.rng.integers(0, self.ttl_jitter + 1))

            if bid_ticks_L not in buy_ticks:
                bid = bid_ticks_L * tick
                oidb = self.model.market.place_limit(self.unique_id, "buy", bid, q, ttl=ttl)
                if oidb is not None:
                    self.live.append(oidb)

            if ask_ticks_L not in sell_ticks:
                ask = ask_ticks_L * tick
                oida = self.model.market.place_limit(self.unique_id, "sell", ask, q, ttl=ttl)
                if oida is not None:
                    self.live.append(oida)

