"""
Flask backend with WebSocket support for real-time simulation streaming.
"""
import json
import time
import threading
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from models.abm_model import UnifiedABMModel
from options.pricing import PricingModel

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global simulation state
simulation_state = {
    'running': False,
    'paused': False,
    'model': None,
    'config': None,
    'step_delay': 0.1  # No delay between steps - maximum speed
}

def get_step_data(model):
    """Extract all relevant data from a simulation step."""
    step = int(model.market.book.t)
    
    # Spot market data
    spot_data = {
        'price': float(model.current_price),
        'best_bid': float(model.market.book.best_bid()) if model.market.book.best_bid() is not None else None,
        'best_ask': float(model.market.book.best_ask()) if model.market.book.best_ask() is not None else None,
        'spread': float(model.spread_log[-1]) if model.spread_log else 0.0,
        'depth_bid': float(model.depth_bid_log[-1]) if model.depth_bid_log else 0.0,
        'depth_ask': float(model.depth_ask_log[-1]) if model.depth_ask_log else 0.0,
        'imbalance': float(model.imbalance_log[-1]) if model.imbalance_log else 0.0,
        'volatility': float(model.volatility),
        'volume': float(model.volume_log[-1]) if model.volume_log else 0.0,
        'trades': int(model.trade_count_log[-1]) if model.trade_count_log else 0,
    }
    
    # Regime data
    regime_data = {
        'regime': int(model.regime),
        'regime_label': 'STRESS' if model.regime == 1 else 'CALM'
    }
    
    # Options market data
    options_data = {}
    if model.enable_options and model.options_market:
        for contract_id, contract in model.options_market.contracts.items():
            spread = model.options_market.get_spread(contract_id)
            depth = model.options_market.get_depth(contract_id)
            mid_price = model.options_market.get_mid_price(contract_id, fallback=0.0)
            
            options_data[f'contract_{contract_id}'] = {
                'contract_id': contract_id,
                'type': contract.option_type,
                'strike': float(contract.strike),
                'maturity': float(contract.maturity),
                'spread': float(spread) if spread is not None else None,
                'depth_bid': float(depth[0]) if depth else 0.0,
                'depth_ask': float(depth[1]) if depth else 0.0,
                'mid_price': float(mid_price) if mid_price else None,
            }
    
    # Model distribution
    model_distribution = {}
    if model.enable_options and len(model.dealer_model_distribution_log) > 0:
        dist = model.dealer_model_distribution_log[-1]
        model_distribution = {
            'BS': float(dist.get(PricingModel.BS, 0.0)),
            'TFBS': float(dist.get(PricingModel.TFBS, 0.0)),
            'HESTON': float(dist.get(PricingModel.HESTON, 0.0)),
        }
    
    # Hedging errors
    hedge_errors = {}
    if model.enable_options and hasattr(model, 'dealer_hedge_errors_by_model'):
        hedge_errors = {
            'BS': float(model.dealer_hedge_errors_by_model.get(PricingModel.BS, [0.0])[-1]) if model.dealer_hedge_errors_by_model.get(PricingModel.BS) else 0.0,
            'TFBS': float(model.dealer_hedge_errors_by_model.get(PricingModel.TFBS, [0.0])[-1]) if model.dealer_hedge_errors_by_model.get(PricingModel.TFBS) else 0.0,
            'HESTON': float(model.dealer_hedge_errors_by_model.get(PricingModel.HESTON, [0.0])[-1]) if model.dealer_hedge_errors_by_model.get(PricingModel.HESTON) else 0.0,
        }
    
    # Send only current step data (incremental updates)
    # Client will accumulate the data
    current_step_idx = len(model.regime_log) - 1
    
    time_series = {
        'current_step': current_step_idx,
        'price': float(model.current_price),
        'spread': float(model.spread_log[-1]) if model.spread_log else 0.0,
        'regime': int(model.regime),
        'volatility': float(model.volatility),
        'volume': float(model.volume_log[-1]) if model.volume_log else 0.0,
        'imbalance': float(model.imbalance_log[-1]) if model.imbalance_log else 0.0,
    }
    
    # Options current values
    if model.enable_options:
        for contract in model.option_contracts:
            cid = contract.contract_id if hasattr(contract, 'contract_id') else contract
            if cid in model.option_spreads_log and len(model.option_spreads_log[cid]) > 0:
                spread_val = model.option_spreads_log[cid][-1]
                time_series[f'option_spread_{cid}'] = float(spread_val) if spread_val == spread_val else None
    
    # Hedging errors current values
    if model.enable_options and hasattr(model, 'dealer_hedge_errors_by_model'):
        for model_type in [PricingModel.BS, PricingModel.TFBS, PricingModel.HESTON]:
            errors = model.dealer_hedge_errors_by_model.get(model_type, [])
            if errors:
                model_name = 'BS' if model_type == PricingModel.BS else ('TFBS' if model_type == PricingModel.TFBS else 'HESTON')
                time_series[f'hedge_error_{model_name}'] = float(errors[-1])
    
    # Model distribution current values
    if model.enable_options and len(model.dealer_model_distribution_log) > 0:
        dist = model.dealer_model_distribution_log[-1]
        time_series['model_dist_BS'] = float(dist.get(PricingModel.BS, 0.0))
        time_series['model_dist_TFBS'] = float(dist.get(PricingModel.TFBS, 0.0))
        time_series['model_dist_HESTON'] = float(dist.get(PricingModel.HESTON, 0.0))
    
    # Spot market trades - get recent trades (last 50 trades or from last 10 steps)
    spot_trades = []
    if hasattr(model.market, 'book') and hasattr(model.market.book, 'trades'):
        # Get recent trades (from last 10 steps or last 50 trades, whichever is smaller)
        recent_trades = model.market.book.trades[-50:] if len(model.market.book.trades) > 50 else model.market.book.trades
        min_step = max(0, step - 10)
        for trade in recent_trades:
            trade_step, price, qty, passive_id, aggressive_id = trade
            if trade_step >= min_step:  # Include trades from recent steps
                spot_trades.append({
                    'step': int(trade_step),
                    'price': float(price),
                    'qty': int(qty),
                    'passive_id': int(passive_id),
                    'aggressive_id': int(aggressive_id)
                })
    
    # Options market trades - get recent trades from all contracts
    options_trades = []
    if model.enable_options and model.options_market:
        min_step = max(0, step - 10)
        for contract_id, lob in model.options_market.lobs.items():
            if hasattr(lob, 'trades'):
                # Get recent trades (last 50 per contract)
                recent_trades = lob.trades[-50:] if len(lob.trades) > 50 else lob.trades
                for trade in recent_trades:
                    trade_step, price, qty, passive_id, aggressive_id = trade
                    if trade_step >= min_step:  # Include trades from recent steps
                        options_trades.append({
                            'step': int(trade_step),
                            'contract_id': int(contract_id),
                            'price': float(price),
                            'qty': int(qty),
                            'passive_id': int(passive_id),
                            'aggressive_id': int(aggressive_id)
                        })
    
    return {
        'step': step,
        'spot': spot_data,
        'regime': regime_data,
        'options': options_data,
        'model_distribution': model_distribution,
        'hedge_errors': hedge_errors,
        'time_series': time_series,
        'hyperparameters': get_hyperparameters(model),
        'spot_trades': spot_trades,
        'options_trades': options_trades
    }

def get_hyperparameters(model):
    """Extract hyperparameters from model."""
    # Count agents by type more reliably
    def count_agents_by_type(agents_list, class_name):
        count = 0
        for agent in agents_list:
            agent_class = agent.__class__.__name__ if hasattr(agent, '__class__') else str(type(agent))
            if class_name in agent_class:
                count += 1
        return count
    
    return {
        'S0': float(model.market.prices[0]) if model.market.prices else 100.0,
        'dt': float(model.dt),
        'steps': int(model.steps_n),
        'n_fund': count_agents_by_type(model.agents_list, 'Fundamentalist'),
        'n_chart': count_agents_by_type(model.agents_list, 'Chartist'),
        'n_mm': count_agents_by_type(model.agents_list, 'MarketMaker'),
        'n_noise': count_agents_by_type(model.agents_list, 'NoiseTrader'),
        'tick_size': float(model.tick_size),
        'enable_options': bool(model.enable_options),
        'n_option_dealers': count_agents_by_type(model.agents_list, 'OptionDealer') if model.enable_options else 0,
        'n_option_takers': count_agents_by_type(model.agents_list, 'OptionTaker') if model.enable_options else 0,
        'n_option_contracts': len(model.option_contracts) if model.enable_options else 0,
        'p01': float(model.p01),
        'p10': float(model.p10),
        'shock_rate': float(model.shock_rate),
        'regime': int(model.regime),
    }

def run_simulation():
    """Run simulation step by step and emit data via WebSocket."""
    global simulation_state
    
    step_count = 0
    last_log_time = time.time()
    
    while simulation_state['running']:
        if simulation_state['paused']:
            time.sleep(0.1)
            continue
        
        try:
            model = simulation_state['model']
            if model is None:
                break
            
            step_start_time = time.time()
            
            # Run one step with timeout protection
            try:
                model.step()
                step_count += 1
            except ValueError as e:
                if "NaN" in str(e) or "cannot convert float NaN" in str(e):
                    print(f"NaN error in simulation step {step_count}: {e}")
                    socketio.emit('simulation_error', {'error': f'NaN error: {str(e)}'})
                    simulation_state['running'] = False
                    break
                raise
            except Exception as e:
                print(f"Error in model.step() at step {step_count}: {e}")
                import traceback
                traceback.print_exc()
                socketio.emit('simulation_error', {'error': f'Step error: {str(e)}'})
                simulation_state['running'] = False
                break
            
            # Get step data
            try:
                step_data = get_step_data(model)
            except Exception as e:
                print(f"Error getting step data at step {step_count}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Emit to all connected clients EVERY STEP
            try:
                socketio.emit('simulation_step', step_data)
            except Exception as e:
                print(f"Error emitting step data at step {step_count}: {e}")
                import traceback
                traceback.print_exc()
            
            # Check if simulation is complete
            # Note: model.market.book.t starts at 0 and increments after each step
            # So for steps_n=200, t will go from 0 to 199 (200 steps total)
            # We stop when t >= steps_n (i.e., when t reaches 200, meaning we've done 200 steps)
            if model.market.book.t >= model.steps_n:
                print(f"Simulation complete: t={model.market.book.t}, steps_n={model.steps_n}, step_count={step_count}")
                simulation_state['running'] = False
                try:
                    socketio.emit('simulation_step', step_data)
                    socketio.emit('simulation_complete', {'message': f'Simulation completed: {step_count} steps executed, t={model.market.book.t}'})
                except:
                    pass
                break
            
            
            # Delay between steps (can be 0 for maximum speed)
            if simulation_state['step_delay'] > 0:
                time.sleep(simulation_state['step_delay'])
            
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
            simulation_state['running'] = False
            break
        except Exception as e:
            import traceback
            print(f"Error in simulation at step {step_count}: {e}")
            traceback.print_exc()
            try:
                socketio.emit('simulation_error', {'error': str(e)})
            except:
                pass
            simulation_state['running'] = False
            break
    

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'status': 'connected'})
    
    # Send current state if simulation is running
    if simulation_state['model'] is not None and simulation_state['running']:
        try:
            step_data = get_step_data(simulation_state['model'])
            emit('simulation_step', step_data)
        except Exception as e:
            print(f"Error sending initial state: {e}")
            import traceback
            traceback.print_exc()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    pass

@socketio.on('start_simulation')
def handle_start_simulation(config):
    """Start a new simulation."""
    global simulation_state
    
    if simulation_state['running']:
        emit('error', {'message': 'Simulation already running'})
        return
    
    try:
        # Default configuration
        default_config = {
            "S0": 100.0,
            "dt": 0.001,
            "steps": 500,  # Reduced for faster testing
            "n_fund": 10,
            "n_chart": 10,
            "n_mm": 3,
            "n_noise": 50,
            "fundamental_price": 100.0,
            "tick_size": 0.1,
            "enable_options": True,
            "n_option_dealers": 3,  # Reduced for faster computation
            "n_option_contracts": 3,
            "n_option_takers": 5,  # Reduced for faster computation
            "enable_model_switching": True,
            "dealer_model_distribution": {
                "BS": 0.5,  # More BS (faster) for testing
                "TFBS": 0.25,
                "HESTON": 0.25
            },
            "p01": 0.005,
            "p10": 0.03,
            "shock_rate": 0.003,
        }
        
        # Merge with provided config
        cfg = {**default_config, **config}
        
        # Convert dealer_model_distribution to PricingModel enum
        if 'dealer_model_distribution' in cfg and isinstance(cfg['dealer_model_distribution'], dict):
            dist = {}
            for k, v in cfg['dealer_model_distribution'].items():
                if k == 'BS':
                    dist[PricingModel.BS] = v
                elif k == 'TFBS':
                    dist[PricingModel.TFBS] = v
                elif k == 'HESTON':
                    dist[PricingModel.HESTON] = v
            cfg['dealer_model_distribution'] = dist
        
        # Create model
        import time
        seed = int(time.time() * 1000000) % (2**31)
        
        try:
            model = UnifiedABMModel(seed=seed, **cfg)
        except Exception as e:
            print(f"Error creating model: {e}")
            import traceback
            traceback.print_exc()
            emit('error', {'message': f'Error creating model: {str(e)}'})
            return
        
        simulation_state['model'] = model
        simulation_state['config'] = cfg
        simulation_state['running'] = True
        simulation_state['paused'] = False
        
        # Start simulation in background thread
        thread = threading.Thread(target=run_simulation, daemon=True)
        thread.start()
        
        emit('simulation_started', {'message': 'Simulation started'})
        
    except Exception as e:
        emit('error', {'message': f'Error starting simulation: {str(e)}'})

@socketio.on('stop_simulation')
def handle_stop_simulation():
    """Stop the simulation."""
    global simulation_state
    simulation_state['running'] = False
    emit('simulation_stopped', {'message': 'Simulation stopped'})

@socketio.on('pause_simulation')
def handle_pause_simulation():
    """Pause/resume the simulation."""
    global simulation_state
    simulation_state['paused'] = not simulation_state['paused']
    status = 'paused' if simulation_state['paused'] else 'resumed'
    emit('simulation_paused', {'status': status})

@socketio.on('set_step_delay')
def handle_set_step_delay(data):
    """Set delay between simulation steps."""
    global simulation_state
    delay = float(data.get('delay', 0.01))
    simulation_state['step_delay'] = max(0.0, min(1.0, delay))  # Clamp between 0 and 1 second
    emit('step_delay_set', {'delay': simulation_state['step_delay']})

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5001))  # Default to 5001, can be overridden by env var
    socketio.run(app, debug=True, host='0.0.0.0', port=PORT)

