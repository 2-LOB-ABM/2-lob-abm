#!/usr/bin/env python3
"""
Quick start script for the real-time simulation dashboard.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

if __name__ == '__main__':
    import os
    from backend.app import app, socketio
    
    PORT = int(os.environ.get('PORT', 5001))  # Default to 5001, can be overridden by env var
    
    print("=" * 60)
    print("ABM Simulation Dashboard")
    print("=" * 60)
    print(f"Starting server on http://localhost:{PORT}")
    print(f"Open your browser and navigate to http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    socketio.run(app, debug=True, host='0.0.0.0', port=PORT)

