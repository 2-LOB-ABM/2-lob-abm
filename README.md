# Unified Two-LOB Agent-Based Market with Options

A comprehensive agent-based market simulation (ABM) that models realistic microstructure with a limit order book (LOB) and an explicit options market where dealers use different pricing models (Black-Scholes, Time-Fractional Black-Scholes, Heston) and hedge their positions in the underlying market.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Detailed Logic](#detailed-logic)
4. [Probabilities and Parameters](#probabilities-and-parameters)
5. [Regime Switching Mechanism](#regime-switching-mechanism)
6. [Options Market](#options-market)
7. [Model Switching (H1)](#model-switching-h1)
8. [Usage Guide](#usage-guide)
9. [Configuration Parameters](#configuration-parameters)
10. [Experiments and Hypotheses](#experiments-and-hypotheses)

---

## Project Overview

This project implements a unified agent-based market model that combines:

1. **Underlying Market**: A realistic limit order book (LOB) with price-time priority, TTL expiration, and multiple agent types
2. **Options Market**: Separate LOB for each option contract with dealers using different pricing models
3. **Hedging Feedback**: Option dealers hedge positions in the underlying market, creating feedback loops
4. **Regime Switching**: Market alternates between calm and stress regimes based on endogenous indicators
5. **Model Switching**: Dealers adaptively switch pricing models based on hedging performance (H1)

---

## Architecture

### Core Components

```
.
├── core/                    # Core market infrastructure
│   ├── lob.py              # Limit Order Book implementation
│   └── market.py            # Market wrapper with price tracking
├── agents/                  # Agent classes
│   ├── base_agents.py      # Underlying market agents
│   └── option_dealers.py   # Option dealers with hedging
├── options/                 # Options market
│   ├── pricing.py          # Pricing models (BS/TFBS/Heston)
│   └── market.py           # Options market with LOB
├── models/                  # ABM orchestration
│   └── abm_model.py        # Unified ABM model
├── experiments/             # Hypothesis testing
│   └── hypotheses.py       # H1-H4 experiments
└── utils/                   # Utilities
    └── analysis.py         # Analysis functions
```

### Data Flow

1. **Market Step**: Each simulation step processes multiple micro-events (determined by Hawkes process)
2. **Agent Actions**: Agents place limit/market orders based on their strategies
3. **Order Matching**: LOB matches orders using price-time priority
4. **Regime Update**: Market regime (calm/stress) updates based on indicators
5. **Options Trading**: Option dealers quote prices, takers hit the book
6. **Hedging**: Dealers hedge delta exposure in underlying market
7. **Model Switching**: Dealers evaluate and switch pricing models (if enabled)

---

## Detailed Logic

### Limit Order Book (LOB) Mechanics

**Location**: `core/lob.py`

The LOB implements a realistic order book with the following features:

#### Order Structure
- **Order ID**: Unique identifier
- **Agent ID**: Which agent placed the order
- **Side**: "buy" or "sell"
- **Quantity**: Integer quantity
- **Price Ticks**: Price in ticks (price = ticks × tick_size)
- **Time**: Current simulation step
- **TTL**: Time-to-live (orders expire after TTL steps)

#### Order Matching Logic

1. **Price-Time Priority**:
   - Buy orders sorted by price (highest first), then time
   - Sell orders sorted by price (lowest first), then time
   - Matching occurs when buy price ≥ sell price

2. **Matching Algorithm** (`_match_incoming`):
   - Incoming order checks best opposite side price
   - If price compatible, matches against resting orders
   - Trades execute at resting order price (price-time priority)
   - Partial fills supported (order quantity reduced until exhausted)

3. **Order Expiration**:
   - Orders stored in `expire_buckets` by expiration time
   - Each step, orders expiring at current time are removed
   - Expired orders don't match and are removed from book

#### Order Types

1. **Limit Orders** (`add_limit`):
   - Specify exact price
   - Rest in book until matched or expired
   - Can match immediately if price compatible

2. **Market Orders** (`add_market`):
   - No price specified (price_ticks = None)
   - Match immediately against best opposite side
   - Execute at best available price

#### Order Book State

- **Bid Side**: Dictionary mapping price_ticks → deque of order IDs
- **Ask Side**: Dictionary mapping price_ticks → deque of order IDs
- **Sorted Price Lists**: `bid_prices` (ascending), `ask_prices` (ascending)
- **Order Registry**: Dictionary mapping order_id → Order object

---

### Agent Behaviors

**Location**: `agents/base_agents.py`

#### 1. Fundamentalist

**Strategy**: Trade based on deviation from fundamental price

**Logic**:
```python
signal = strength * (F - S) / S + noise_scale * ε
```
- `F`: Fundamental price (model parameter)
- `S`: Current market price
- `strength`: Trading strength (default: 0.5)
- `noise_scale`: Noise amplitude (default: 0.15)
- `ε`: Standard normal random variable

**Order Placement**:
- Quantity: `min(max_qty, round(abs(signal) * max_qty))`
- Side: "buy" if signal > 0, "sell" otherwise
- Type: Always market orders

**Probability**: Always trades (no probability threshold)

#### 2. Chartist

**Strategy**: Momentum/trend-following with volatility sensitivity

**Logic**:
```python
signal = strength * mom * (1.0 + vol_sens * v) + noise_scale * ε
```
- `mom`: Recent momentum (mean log returns over `mom_window`, default: 20 steps)
- `v`: Recent volatility (std of log returns over `vol_window`, default: 20 steps)
- `vol_sens`: Volatility sensitivity multiplier (default: 2.0)
- `strength`: Trading strength (default: 0.6)

**Order Placement**:
- Quantity: `min(max_qty, round(abs(signal) * max_qty))`
- Side: "buy" if signal > 0, "sell" otherwise
- Type: Always market orders

**Probability**: Always trades (no probability threshold)

#### 3. Noise Trader

**Strategy**: Random trading with regime-dependent behavior

**Logic**:
- **Trading Probability**: `prob` (default: 0.2 per step)
- **Sell Bias**: `sell_bias` (default: 0.5, increases by 0.20 in stress regime)
- **Quantity**: Pareto-distributed: `pareto_int(rng, xm=1.0, alpha=1.6, cap=200)`

**Pareto Distribution**:
```python
x = xm / (u^(1/alpha))
```
- `xm`: Minimum value (1.0)
- `alpha`: Shape parameter (1.6)
- `cap`: Maximum value (200)
- `u`: Uniform random [0,1]

**Order Type Selection**:
- **Limit Order Probability**: 0.85 in calm, 0.35 in stress
- **Market Order**: Otherwise
- **Limit Price**: `mid ± (limit_offset_ticks + regime) × tick_size`
  - `limit_offset_ticks`: Default 2
  - In stress, offset increases by 1 tick

**Regime Dependence**:
- Sell bias increases in stress: `min(0.95, sell_bias + 0.20 × regime)`
- More market orders in stress (lower limit_prob)

#### 4. Market Maker

**Strategy**: Provide liquidity by quoting bid/ask spreads

**Logic**:
- Maintains quotes at multiple price levels
- Adjusts quotes when mid price moves significantly
- Reduces liquidity in stress regime

**Quote Update Conditions** (`need_requote`):
1. First time quoting (`last_mid_ticks` is None)
2. Mid price moved ≥ `requote_move_ticks` (default: 5 ticks, reduced by 1 in stress)
3. Best bid or ask is None (empty book)
4. Depth at best is zero
5. Number of live quotes < 2 × desired_levels

**Quote Structure**:
- **Spread**: `base_spread_ticks + (3 if stress else 0)`
- **Levels**: `max(1, levels - (2 if stress else 0))` (default: 5 levels)
- **Size**: `max(1, size // (2 if stress else 1))` (default: 15)
- **TTL**: `ttl + random(0, ttl_jitter)` (default: 10, jitter: 5)

**Quote Placement**:
- Bid levels: `mid_ticks - spread - L` for L = 0, 1, ..., levels-1
- Ask levels: `mid_ticks + spread + L` for L = 0, 1, ..., levels-1
- Quantity per level: `base_size // (L + 1)` (decreasing with distance)

**Market Maker Reaction**:
- Market makers react to trades with probability `mm_react_prob` (default: 0.6)
- Latency: `mm_latency_events` steps (default: 2) before reaction
- Reaction tracked via `_mm_pending` counter

---

### Hawkes Process

**Location**: `models/abm_model.py` (step method)

The Hawkes process determines the number of micro-events per simulation step.

#### Intensity Function

```python
λ(t) = μ + α × H(t)
```

- `μ`: Base intensity (default: 200.0)
- `α`: Self-excitation parameter (default: 0.5)
- `H(t)`: Decaying history of past events

#### History Decay

```python
H(t+1) = exp(-β × dt) × H(t) + n_events
```

- `β`: Decay rate (default: 5.0)
- `dt`: Time step (default: 0.001)
- `n_events`: Number of events in current step

#### Regime Scaling

```python
λ_reg(t) = λ(t) × (base_scale / n_events_calm)
```

- `base_scale`: `n_events_calm` (400) in calm, `n_events_stress` (1200) in stress
- This scales intensity by regime

#### Event Generation

```python
Nn ~ Poisson(λ_reg × dt)
n_events = min(max_events, max(1, Nn))
```

- Poisson distribution with mean `λ_reg × dt`
- Capped at `max_events` (default: 3000)
- Minimum 1 event per step

**Interpretation**:
- Each event allows one agent to act
- More events → more trading activity
- Stress regime → higher intensity → more events
- Past events increase future intensity (self-excitation)

---

## Probabilities and Parameters

### Agent Probabilities

| Agent Type | Probability | Description |
|------------|-------------|-------------|
| Fundamentalist | 1.0 | Always trades when selected |
| Chartist | 1.0 | Always trades when selected |
| Noise Trader | `prob` (default: 0.2) | Probability of trading per step |
| Market Maker | `mm_react_prob` (default: 0.6) | Probability of reacting to trades |
| Option Taker | `option_taker_trade_prob` (default: 0.01) | Probability of trading per step |

### Order Type Probabilities (Noise Trader)

| Regime | Limit Order | Market Order |
|--------|-------------|--------------|
| Calm | 0.85 | 0.15 |
| Stress | 0.35 | 0.65 |

### Regime Transition Probabilities

**Calm → Stress**:
```python
p = min(1.0, p01 × (1.0 + shock_trigger × 0.5))
```
- `p01`: Base probability (default: 0.02, reduced to 0.005 in examples)
- `shock_trigger`: Weighted sum of stress indicators (see Regime Switching)

**Stress → Calm**:
```python
recovery_factor = 1.0 - min(0.5, shock_trigger × 0.3)
p = p10 × recovery_factor
```
- `p10`: Base probability (default: 0.10, reduced to 0.03 in examples)
- Recovery slower if stress indicators persist

### Exogenous Shock Probability

**Shock Generation**:
```python
if random() < shock_rate:
    generate_shock()
```
- `shock_rate`: Default 0.01, reduced to 0.003 in examples
- Shocks are metaorders (large sell/buy orders)

**Shock Duration**:
- Base duration: Random(40, 80) steps
- In stress: Duration × 2
- Intensity: Random(5, 12) orders per step, × 2 in stress

### Model Switching Probabilities (H1)

**Switching Conditions** (see Model Switching section for details):

| Condition | Probability | Threshold |
|-----------|-------------|-----------|
| Large error | 1.0 (deterministic) | mean_error > 0.03 |
| Medium error | 0.8 | mean_error > 0.01 |
| Small error + loss | 0.6 | mean_error > 0.005 AND mean_pnl < -0.005 |
| Strong increasing trend | 1.0 (deterministic) | trend_strength > 0.3 |
| Moderate trend | 0.8 | trend_strength > 0.15 |
| Weak trend | 0.5 | trend_strength > 0.05 |

**Cooldown**: Minimum 20 steps between switches

---

## Regime Switching Mechanism

**Location**: `models/abm_model.py` (`update_regime` method)

The market switches between two regimes:
- **Regime 0 (Calm)**: Normal market conditions
- **Regime 1 (Stress)**: Elevated volatility, wider spreads, higher activity

### Stress Indicators

Three indicators are computed each step:

1. **Return Flag** (`r_flag`):
```python
r_flag = 1.0 if last_r < -theta_r × σ × sqrt(dt) else 0.0
```
- `last_r`: Last log return
- `theta_r`: Threshold multiplier (default: 3.0)
- `σ`: Realized volatility (50-step window)

2. **Spread Flag** (`s_flag`):
```python
s_flag = 1.0 if spread > theta_s else 0.0
```
- `spread`: Current bid-ask spread
- `theta_s`: Threshold (default: 3.0 × tick_size)

3. **Imbalance Flag** (`I_flag`):
```python
I_flag = 1.0 if |imbalance| > theta_I else 0.0
```
- `imbalance = (depth_bid - depth_ask) / (depth_bid + depth_ask)`
- `theta_I`: Threshold (default: 0.5)

### Shock Trigger

```python
shock_trigger = xi_wr × r_flag + xi_ws × s_flag + xi_wI × I_flag
```

- `xi_wr`: Weight for return flag (default: 1.0)
- `xi_ws`: Weight for spread flag (default: 1.0)
- `xi_wI`: Weight for imbalance flag (default: 1.0)

### Transition Probabilities

**Calm → Stress**:
```python
p = min(1.0, p01 × (1.0 + shock_trigger × 0.5))
if random() < p:
    regime = 1
```

**Stress → Calm**:
```python
recovery_factor = 1.0 - min(0.5, shock_trigger × 0.3)
p = p10 × recovery_factor
if random() < p:
    regime = 0
```

### Regime Effects

**On Hawkes Intensity**:
- Calm: `base_scale = n_events_calm` (400)
- Stress: `base_scale = n_events_stress` (1200)

**On Market Makers**:
- Spread increases by 3 ticks in stress
- Number of levels decreases by 2 in stress
- Quote size halves in stress
- Requote threshold decreases by 1 tick in stress

**On Noise Traders**:
- Sell bias increases by 0.20 in stress
- Limit order probability decreases (0.85 → 0.35)
- Limit offset increases by 1 tick in stress

**On Exogenous Shocks**:
- Shock duration doubles in stress
- Shock intensity doubles in stress

---

## Options Market

**Location**: `options/market.py`, `options/pricing.py`, `agents/option_dealers.py`

### Option Contracts

Each contract has:
- **Contract ID**: Unique identifier
- **Option Type**: "call" or "put"
- **Strike**: Strike price
- **Maturity**: Time to expiration (decreases each step)
- **Underlying Price**: Reference price

**Default Contracts** (if `n_option_contracts = 3`):
1. ATM Call: Strike = S0
2. ATM Put: Strike = S0
3. OTM Put: Strike = 0.9 × S0

### Options LOB

Each contract has its own `LimitOrderBook` instance:
- Same matching logic as underlying LOB
- Separate order book per contract
- Contracts expire when `maturity <= 0`

### Option Dealers

**Location**: `agents/option_dealers.py`

#### Pricing Models

1. **Black-Scholes (BS)**:
   - Analytical formulas for price, delta, gamma, vega
   - Assumes constant volatility
   - Fast computation

2. **Time-Fractional Black-Scholes (TFBS)**:
   - Memory parameter `α` (default: 0.85, range: 0.7-1.0)
   - `α = 1` recovers standard BS
   - Lower `α` → more memory/fat tails
   - Monte Carlo pricing (default: 5000 paths)
   - Greeks via finite differences

3. **Heston Stochastic Volatility**:
   - Stochastic variance process
   - Parameters:
     - `kappa`: Mean reversion speed (default: 2.0)
     - `theta`: Long-term variance (default: 0.04)
     - `sigma_v`: Volatility of volatility (default: 0.3)
     - `rho`: Correlation asset-variance (default: -0.7)
   - Monte Carlo pricing (default: 5000 paths, 100 steps)
   - Greeks via finite differences

#### Reservation Price

```python
res_price = model_price - inventory_adjustment
inventory_adjustment = inventory_risk_aversion × position × price × 0.01
```

- `model_price`: Price from dealer's pricing model
- `inventory_risk_aversion`: Default 0.1
- If long inventory, adjust price down (discount to sell)
- If short inventory, adjust price up (premium to buy)

#### Quote Placement

```python
spread = res_price × base_spread_pct  # Default: 2%
bid_price = res_price - spread / 2
ask_price = res_price + spread / 2
```

- Quotes placed as limit orders
- TTL: 10 steps
- Size: `quote_size` (default: 10 contracts)

#### Greeks Calculation

**Delta**:
- Call: `N(d1)` (BS) or finite difference (TFBS/Heston)
- Put: `N(d1) - 1` (BS) or finite difference

**Gamma**:
- Second derivative w.r.t. price
- Same for call and put

**Vega**:
- Sensitivity to volatility
- Same for call and put

### Hedging Mechanism

**Location**: `agents/option_dealers.py` (`_hedge_position` method)

#### Delta Calculation

```python
total_delta = sum(position × delta for each contract)
delta_to_hedge = total_delta - hedge_position
```

- `position`: Net option position (positive = long, negative = short)
- `delta`: Option delta from dealer's model
- `hedge_position`: Current hedge in underlying market

#### Hedge Execution

```python
if abs(delta_to_hedge) > 0.01:
    qty = int(abs(delta_to_hedge))
    side = "buy" if delta_to_hedge > 0 else "sell"
    place_market_order(side, qty)
    hedge_position += delta_to_hedge
```

- Hedge frequency: Every `hedge_frequency` steps (default: 1 = every step)
- Threshold: 0.01 to avoid micro-trades

#### Hedging Error Tracking

**Error Calculation**:
```python
hedge_pnl = hedge_position × price_change
option_pnl = current_option_value - last_option_value
hedge_error = |option_pnl - hedge_pnl|
```

- Tracks how well hedge matches option PnL
- Accumulated in `hedge_errors` list
- Used for model switching evaluation

### Option Takers

**Location**: `agents/option_dealers.py` (`OptionTaker` class)

**Trading Logic**:
- Probability: `option_taker_trade_prob` (default: 0.01 per step)
- Randomly selects contract
- Side: Random, with put bias in stress (0.6 vs 0.5)
- Quantity: Random(1, max_qty) (default: max_qty = 5)
- Order type: Market order (hits dealer quotes)

---

## Model Switching (H1)

**Location**: `agents/option_dealers.py` (`_evaluate_model_performance` method)

Dealers can switch between pricing models based on hedging performance.

### Performance Evaluation

**Error Metrics**:
- `mean_error`: Average hedging error over `switching_window` (default: 50 steps)
- `max_error`: Maximum error in window
- `mean_pnl`: Average PnL (option_pnl - hedge_pnl)
- `trend_strength`: Relative increase in errors over `trend_window` (default: 5 steps)

**Error Trend Detection**:
```python
if errors[i] > errors[i-1] for most i in trend_window:
    trend_increasing = True
    trend_strength = (errors[-1] - errors[0]) / |errors[0]|
```

### Switching Conditions

1. **Large Error** (deterministic):
   - `mean_error > 0.03` → Switch immediately

2. **Medium Error** (probabilistic):
   - `mean_error > 0.01` → Switch with probability 0.8

3. **Small Error + Loss** (probabilistic):
   - `mean_error > 0.005 AND mean_pnl < -0.005` → Switch with probability 0.6

4. **Increasing Trend**:
   - Strong (`trend_strength > 0.3`): Switch deterministically
   - Moderate (`trend_strength > 0.15`): Switch with probability 0.8
   - Weak (`trend_strength > 0.05`): Switch with probability 0.5

### Model Selection

**Comparison Logic**:
- Dealer tracks errors for ALL models (not just current)
- Calculates `model_avg_errors` for each model
- Chooses model with lowest average error
- If current model is best, don't switch

**Cooldown**:
- Minimum `switch_cooldown` steps (default: 20) between switches
- Prevents rapid oscillation

### Model Error Calculation

**For Current Model**:
- Uses actual hedging errors from trades

**For Other Models**:
- Calculates theoretical error using model's Greeks
- Error ≈ `|position × gamma × price_change²|`
- Models with higher gamma have larger errors when price moves

### Switching Effects

**After Switch**:
- Pricer reinitialized with model-specific parameters
- Performance history kept (last 25% of errors)
- Error trend reset
- Last hedge price reset (recalculate on next hedge)

---

## Usage Guide

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install backend dependencies (for dashboard)
cd backend
pip install -r requirements.txt
cd ..
```

### Basic Simulation

```python
from models.abm_model import UnifiedABMModel

# Configuration
cfg = {
    "S0": 100.0,                    # Initial price
    "dt": 0.001,                    # Time step
    "steps": 1000,                  # Number of simulation steps
    "n_fund": 10,                   # Number of fundamentalists
    "n_chart": 10,                  # Number of chartists
    "n_mm": 3,                      # Number of market makers
    "n_noise": 50,                  # Number of noise traders
    "fundamental_price": 100.0,     # Fundamental price
    "tick_size": 0.1,               # Price tick size
    "enable_options": True,         # Enable options market
    "n_option_dealers": 5,          # Number of option dealers
    "n_option_contracts": 3,       # Number of option contracts
    "n_option_takers": 10,          # Number of option takers
    "enable_model_switching": True, # Enable model switching (H1)
}

# Run simulation
model = UnifiedABMModel(seed=42, **cfg)
prices = model.run()

# Access results
print(f"Final price: {prices[-1]}")
print(f"Regime log: {model.regime_log}")
print(f"Model distribution: {model.dealer_model_distribution_log}")
```

### Running Examples

```bash
# Run basic example
python example_usage.py
```

This will:
1. Run a basic simulation with options market
2. Run a model switching demonstration
3. Generate plots showing:
   - Price path
   - Regime timeline
   - Model distribution over time
   - Hedging errors by model

### Running Experiments

```python
from experiments.hypotheses import run_h1_experiment, run_h2_experiment

# H1: Model switching
cfg = {
    "S0": 100.0,
    "dt": 0.001,
    "steps": 2000,
    "enable_options": True,
    "n_option_dealers": 15,
    "enable_model_switching": True,
    # ... other parameters
}

results_h1 = run_h1_experiment(cfg, n_replications=10)

# H2: Heterogeneous vs homogeneous
results_h2 = run_h2_experiment(cfg, n_replications=10)
```

### Running Dashboard

```bash
# Start dashboard server
python run_dashboard.py

# Or directly
cd backend
python app.py
```

Then open `http://localhost:5001` in your browser.

The dashboard provides:
- Real-time price visualization
- Regime indicators
- Option market metrics
- Model distribution charts
- Hedging error tracking

---

## Configuration Parameters

### Market Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `S0` | 100.0 | Initial price |
| `dt` | 0.001 | Time step |
| `steps` | 1000 | Number of simulation steps |
| `tick_size` | 0.1 | Minimum price increment |
| `fundamental_price` | S0 | Fundamental price for fundamentalists |

### Agent Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_fund` | 10 | Number of fundamentalists |
| `n_chart` | 10 | Number of chartists |
| `n_mm` | 3 | Number of market makers |
| `n_noise` | 50 | Number of noise traders |
| `fund_strength` | 0.5 | Fundamentalist trading strength |
| `chart_strength` | 0.6 | Chartist trading strength |
| `chart_vol_sens` | 2.0 | Chartist volatility sensitivity |
| `mom_window` | 20 | Momentum window (steps) |
| `vol_window` | 20 | Volatility window (steps) |
| `agent_noise` | 0.15 | Agent noise scale |

### Market Maker Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_spread_ticks` | 2 | Base spread in ticks |
| `mm_size` | 15 | Quote size |
| `mm_react_prob` | 0.6 | Probability of reacting to trades |
| `mm_latency_events` | 2 | Latency in events before reaction |

### Options Market Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_options` | True | Enable options market |
| `n_option_contracts` | 3 | Number of option contracts |
| `option_tick_size` | 0.01 | Tick size for options |
| `n_option_dealers` | 5 | Number of option dealers |
| `n_option_takers` | 10 | Number of option takers |
| `option_taker_trade_prob` | 0.01 | Probability of taker trading per step |
| `dealer_model_distribution` | {BS: 0.33, TFBS: 0.33, HESTON: 0.34} | Initial model distribution |
| `enable_model_switching` | True | Enable model switching (H1) |
| `switching_window` | 50 | Window for performance evaluation |
| `switching_threshold` | 0.05 | Minimum improvement to switch |

### Regime Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p01` | 0.02 | Transition probability calm → stress |
| `p10` | 0.10 | Transition probability stress → calm |
| `shock_rate` | 0.01 | Exogenous shock frequency |
| `shock_impact` | 8.0 | Shock impact multiplier |
| `n_events_calm` | 400 | Base event rate in calm |
| `n_events_stress` | 1200 | Base event rate in stress |
| `xi_wr` | 1.0 | Weight for return flag |
| `xi_ws` | 1.0 | Weight for spread flag |
| `xi_wI` | 1.0 | Weight for imbalance flag |
| `theta_r` | 3.0 | Return threshold multiplier |
| `theta_s` | 3.0 × tick_size | Spread threshold |
| `theta_I` | 0.5 | Imbalance threshold |

### Hawkes Process Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hawkes_mu` | 200.0 | Base intensity |
| `hawkes_alpha` | 0.5 | Self-excitation parameter |
| `hawkes_beta` | 5.0 | Decay rate |
| `max_events` | 3000 | Maximum events per step |

### Option Dealer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 0.0 | Risk-free rate |
| `base_spread_pct` | 0.02 | Option spread percentage |
| `quote_size` | 10 | Quote size in contracts |
| `hedge_frequency` | 1 | Hedge every N steps |
| `inventory_risk_aversion` | 0.1 | Inventory risk adjustment |

### TFBS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.85 | Memory parameter (0.7-1.0) |
| `n_mc` | 5000 | Monte Carlo paths |

### Heston Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kappa` | 2.0 | Mean reversion speed |
| `theta` | 0.04 | Long-term variance |
| `sigma_v` | 0.3 | Volatility of volatility |
| `rho` | -0.7 | Correlation asset-variance |
| `n_mc` | 5000 | Monte Carlo paths |
| `n_steps` | 100 | Time steps per path |

---

## Experiments and Hypotheses

### H1: Endogenous Model Switching

**Hypothesis**: Dealers endogenously switch between pricing models based on hedging performance. In calm markets, BS dominates; in stress, tail-aware models (TFBS/Heston) gain share.

**Test**: `run_h1_experiment(cfg, n_replications=10)`

**Measures**:
- Model distribution over time
- Switching frequency
- Regime-dependent model preferences
- Performance improvement after switching

**Expected Results**:
- BS share higher in calm regimes
- TFBS/Heston share higher in stress regimes
- Switching frequency correlates with volatility

### H2: Heterogeneous Models Amplify Regimes

**Hypothesis**: Heterogeneous dealer models (BS/TFBS/Heston mix) amplify endogenous volatility regimes compared to homogeneous models, increasing stress episode frequency and volatility clustering.

**Test**: `run_h2_experiment(cfg, n_replications=10)`

**Measures**:
- Stress episode frequency
- Stress episode duration
- Volatility clustering (autocorrelation)
- Regime transition probabilities

**Expected Results**:
- Heterogeneous models → more frequent stress episodes
- Higher volatility autocorrelation
- Longer stress durations

### H3: Adaptive Switching Increases Instability

**Hypothesis**: Adaptive model switching increases volatility amplitude and regime transition frequency compared to fixed strategies.

**Test**: `run_h3_experiment(cfg, n_replications=10)`

**Measures**:
- Volatility amplitude (max - min)
- Regime transition frequency
- Mean and std of volatility

**Expected Results**:
- Adaptive switching → higher volatility amplitude
- More regime transitions
- Higher volatility variance

### H4: Memory ↔ Liquidity

**Hypothesis**: The memory parameter (α) in TFBS is negatively correlated with option market liquidity and positively correlated with extreme price movements.

**Test**: `run_h4_experiment(cfg, alpha_values=[0.7, 0.8, 0.85, 0.9, 1.0], n_replications=10)`

**Measures**:
- Option bid-ask spreads vs α
- Option depth vs α
- Frequency of extreme price moves vs α

**Expected Results**:
- Lower α → wider spreads (lower liquidity)
- Lower α → lower depth
- Lower α → more extreme moves

---

## Output Data

The model logs extensive data:

### Price and Market Data
- `prices`: Price path
- `log_returns`: Log returns
- `spread_log`: Bid-ask spreads
- `depth_bid_log`, `depth_ask_log`: Depth at best
- `imbalance_log`: Order book imbalance
- `volume_log`: Trading volume
- `trade_count_log`: Number of trades per step

### Regime Data
- `regime_log`: Regime at each step (0=calm, 1=stress)
- `n_events_log`: Number of events per step
- `lambda_log`: Hawkes intensity
- `hawkes_H_log`: Hawkes history

### Options Market Data
- `option_spreads_log`: Spreads per contract
- `option_depths_log`: Depths per contract
- `option_trades_log`: Trades per contract

### Model Switching Data
- `dealer_model_distribution_log`: Model distribution over time
- `dealer_hedge_errors_by_model`: Average hedging errors by model

---

## Key Implementation Details

### Order Matching Priority

1. **Price Priority**: Best price first
2. **Time Priority**: Earlier orders at same price first
3. **Immediate Matching**: Market orders match immediately
4. **Partial Fills**: Orders can be partially filled

### Price Discovery

- Mid price: `(best_bid + best_ask) / 2`
- If no bid/ask, uses last trade price
- If no trade, uses previous mid price

### Volatility Calculation

```python
realized_sigma = std(log_returns[-window:]) / sqrt(dt)
```

- Uses rolling window (default: 50 steps)
- Annualized by dividing by `sqrt(dt)`

### Contract Expiration

- Contracts expire when `maturity <= 0`
- Expired contracts removed from market
- Dealers/takers for expired contracts remain (could be cleaned up)

### Random Number Generation

- Uses `numpy.random.default_rng(seed)` for reproducibility
- Each component has separate RNG or uses model's RNG
- Seed can be set for reproducible results

---

## References

- Agent-based Computational Finance
- Heterogeneous Agent Models in Economics and Finance
- Empirical properties of asset returns: stylized facts
- The Price Impact of Order Book Events
- A Dynamic Model of the Limit Order Book
- Black-Scholes option pricing model
- Time-fractional Black-Scholes model
- Heston stochastic volatility model


