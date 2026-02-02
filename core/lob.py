"""
Limit Order Book implementation for both underlying and options markets.
"""
import numpy as np
from dataclasses import dataclass
from collections import deque
from bisect import bisect_left
import itertools


@dataclass
class Order:
    oid: int
    agent_id: int
    side: str
    qty: int
    price_ticks: int | None
    t: int
    ttl: int


class LimitOrderBook:
    """Limit Order Book with price-time priority and TTL expiration."""
    
    def __init__(self, tick_size, rng, max_levels=2000):
        self.tick_size = float(tick_size)
        self.rng = rng
        self.max_levels = int(max_levels)
        self.expire_buckets = {}

        self.bids = {}
        self.asks = {}
        self.bid_prices = []
        self.ask_prices = []

        self.orders = {}
        self._oid = itertools.count(1)

        self.last_trade_price = None
        self.last_trade_ticks = None

        self.trade_happened = False
        self.trades = []
        self.t = 0
        self.trades_in_step = 0
        self.volume_in_step = 0

        # logs
        self.n_limit_in_step = 0
        self.n_market_in_step = 0
        self.n_cancel_in_step = 0
        self.n_expire_in_step = 0
        self.crossed_in_step = 0

    def _insert_price(self, side, p):
        if side == "buy":
            if p in self.bids:
                return
            idx = bisect_left(self.bid_prices, p)
            self.bid_prices.insert(idx, p)
            self.bids[p] = deque()
        else:
            if p in self.asks:
                return
            idx = bisect_left(self.ask_prices, p)
            self.ask_prices.insert(idx, p)
            self.asks[p] = deque()

    def _remove_price_if_empty(self, side, p):
        if side == "buy":
            q = self.bids.get(p)
            if q is not None and len(q) == 0:
                del self.bids[p]
                i = bisect_left(self.bid_prices, p)
                if i < len(self.bid_prices) and self.bid_prices[i] == p:
                    self.bid_prices.pop(i)
        else:
            q = self.asks.get(p)
            if q is not None and len(q) == 0:
                del self.asks[p]
                i = bisect_left(self.ask_prices, p)
                if i < len(self.ask_prices) and self.ask_prices[i] == p:
                    self.ask_prices.pop(i)

    def best_bid_ticks(self):
        if not self.bid_prices:
            return None
        return self.bid_prices[-1]

    def best_ask_ticks(self):
        if not self.ask_prices:
            return None
        return self.ask_prices[0]

    def best_bid(self):
        p = self.best_bid_ticks()
        return None if p is None else p * self.tick_size

    def best_ask(self):
        p = self.best_ask_ticks()
        return None if p is None else p * self.tick_size

    def mid_price(self, fallback):
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None and ba is None:
            return float(fallback)
        if bb is None:
            return float(ba)
        if ba is None:
            return float(bb)
        return 0.5 * (float(bb) + float(ba))

    def spread(self):
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return float(ba - bb)

    def _record_trade(self, price_ticks, qty, passive_agent_id, aggressive_agent_id):
        px = float(price_ticks) * self.tick_size
        self.last_trade_price = px
        self.last_trade_ticks = int(price_ticks)
        self.trades.append((self.t, px, int(qty), int(passive_agent_id), int(aggressive_agent_id)))
        self.trade_happened = True

    def _drop_order(self, oid):
        o = self.orders.pop(oid, None)
        if o is None:
            return False
        if o.price_ticks is None:
            return True

        book = self.bids if o.side == "buy" else self.asks
        prices = self.bid_prices if o.side == "buy" else self.ask_prices

        q = book.get(o.price_ticks)
        if q is not None:
            try:
                q.remove(oid)
            except ValueError:
                pass
            if len(q) == 0:
                del book[o.price_ticks]
                i = bisect_left(prices, o.price_ticks)
                if i < len(prices) and prices[i] == o.price_ticks:
                    prices.pop(i)
        return True

    def cancel(self, oid):
        if oid not in self.orders:
            return False
        self.n_cancel_in_step += 1
        return bool(self._drop_order(oid))
    
    def cancel_one_at_best(self, side):
        if side == "buy":
            p = self.best_bid_ticks()
            if p is None:
                return False
            q = self.bids.get(p)
        else:
            p = self.best_ask_ticks()
            if p is None:
                return False
            q = self.asks.get(p)

        if q is None:
            return False

        while len(q) > 0:
            oid = q[0]
            o = self.orders.get(oid)
            if o is None or o.qty <= 0:
                q.popleft()
                continue
            return bool(self.cancel(oid))

        self._remove_price_if_empty("buy" if side == "buy" else "sell", p)
        return False

    def add_limit(self, agent_id, side, price, qty, ttl=1):
        side = "buy" if side == "buy" else "sell"
        qty = int(qty)
        if qty <= 0:
            return None
                
        self.n_limit_in_step += 1
        price_ticks = int(round(float(price) / self.tick_size))
        oid = next(self._oid)
        o = Order(oid=oid, agent_id=int(agent_id), side=side, qty=qty, price_ticks=price_ticks, t=self.t, ttl=int(ttl))
        if o.ttl > 0:
            self.expire_buckets.setdefault(o.t + o.ttl, []).append(oid)

        self._match_incoming(o)
        if o.qty > 0:
            self._insert_price(side, price_ticks)
            if side == "buy":
                self.bids[price_ticks].append(oid)
            else:
                self.asks[price_ticks].append(oid)
            self.orders[oid] = o
        return oid

    def add_market(self, agent_id, side, qty):
        side = "buy" if side == "buy" else "sell"
        qty = int(qty)
        if qty <= 0:
            return None
        
        self.n_market_in_step += 1
        oid = next(self._oid)
        o = Order(oid=oid, agent_id=int(agent_id), side=side, qty=qty, price_ticks=None, t=self.t, ttl=0)
        if o.ttl > 0:
            self.expire_buckets.setdefault(o.t + o.ttl, []).append(oid)

        self._match_incoming(o)
        return oid

    def _pop_front_order(self, side, price_ticks):
        book = self.bids if side == "buy" else self.asks
        q = book.get(price_ticks)
        while q is not None and len(q) > 0:
            oid = q[0]
            o = self.orders.get(oid)
            if o is None or o.qty <= 0:
                q.popleft()
                continue
            return oid, o
        return None, None

    def _match_incoming(self, incoming):
        while incoming.qty > 0:
            if incoming.side == "buy":
                best_ask = self.best_ask_ticks()
                if best_ask is None:
                    break
                if incoming.price_ticks is not None and incoming.price_ticks < best_ask:
                    break
                oid, resting = self._pop_front_order("sell", best_ask)
                if resting is None:
                    self._remove_price_if_empty("sell", best_ask)
                    continue
                traded = min(incoming.qty, resting.qty)
                incoming.qty -= traded
                resting.qty -= traded
                self._record_trade(best_ask, traded, resting.agent_id, incoming.agent_id)
                if resting.qty == 0:
                    self._drop_order(oid)

            else:
                best_bid = self.best_bid_ticks()
                if best_bid is None:
                    break
                if incoming.price_ticks is not None and incoming.price_ticks > best_bid:
                    break
                oid, resting = self._pop_front_order("buy", best_bid)
                if resting is None:
                    self._remove_price_if_empty("buy", best_bid)
                    continue
                traded = min(incoming.qty, resting.qty)
                incoming.qty -= traded
                resting.qty -= traded
                self._record_trade(best_bid, traded, resting.agent_id, incoming.agent_id)
                if resting.qty == 0:
                    self._drop_order(oid)
            
            self.trades_in_step += 1
            self.volume_in_step += float(traded)

            bb = self.best_bid_ticks()
            ba = self.best_ask_ticks()
            if bb is not None and ba is not None and bb >= ba:
                self.crossed_in_step += 1

    def step_time(self):
        self.t += 1
        bucket = self.expire_buckets.pop(self.t, None)
        if bucket:
            for oid in bucket:
                self.n_expire_in_step += 1
                self._drop_order(oid)

    def reset_step_counters(self):
        self.trades_in_step = 0
        self.volume_in_step = 0.0
        self.n_limit_in_step = 0
        self.n_market_in_step = 0
        self.n_cancel_in_step = 0
        self.n_expire_in_step = 0
        self.crossed_in_step = 0
        self.trade_happened = False

    def depth_at_best(self):
        bb = self.best_bid_ticks()
        ba = self.best_ask_ticks()
        db = self._level_qty("buy", bb) if bb is not None else 0.0
        da = self._level_qty("sell", ba) if ba is not None else 0.0
        return float(db), float(da)

    def _level_qty(self, side, price_ticks):
        book = self.bids if side == "buy" else self.asks
        q = book.get(price_ticks)
        if q is None:
            return 0.0
        s = 0
        for oid in q:
            o = self.orders.get(oid)
            if o is not None and o.qty > 0:
                s += o.qty
        return float(s)
    
    def snapshot_l2(self, depth=10):
        bid_ticks = sorted(self.bid_prices, reverse=True)[:depth]
        ask_ticks = sorted(self.ask_prices)[:depth]

        bid_levels = [(p, self._level_qty("buy", p)) for p in bid_ticks]
        ask_levels = [(p, self._level_qty("sell", p)) for p in ask_ticks]
        return bid_levels, ask_levels

