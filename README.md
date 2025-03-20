# BTC Pattern Finder

A tool to discover similar historical patterns in Bitcoin price movements.

## Overview

BTC Pattern Finder analyzes Bitcoin price data to find historical patterns that match specified timeframes. It uses the Matrix Profile algorithm (via the STUMPY library) to identify the most similar price shapes across Bitcoin's history.

## Features

- Search for patterns across multiple timeframes (1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
- Visualize matches with what happened after each pattern
- Compare current price movements with historical patterns
- Rank matches by similarity score
- View detailed price data for each match
- Access real BTC price data from Binance API (back to 2017)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd btc_pattern_finder
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. The app will open in your browser (default: http://localhost:8501)

3. Interact with the application:
   - Select a timeframe (1m to 1w)
   - Choose recent data or specify a date range
   - Select a pattern on the chart
   - Adjust pattern length and other parameters
   - Click "Find Similar Patterns"

## How It Works

The pattern matching uses a technique called Matrix Profile (implemented by the STUMPY library). It works by:

1. Z-normalizing the price data (removing mean and scaling by standard deviation)
2. Computing sliding window comparisons between your selected pattern and historical data
3. Ranking matches by a distance score (lower means more similar patterns)
4. Filtering out duplicate matches that are too close in time

The z-normalization means we're matching the **shape** of the pattern rather than absolute price levels.


### Data Source

The system uses Binance API to fetch BTCUSDT spot market data:
- Historical data available back to 2017 (Binance's launch)
- Support for timeframes from 1m to 1w
- Automatically handles long time periods by chunking requests
- No API key required for historical data access

## Project Structure

- `app.py` - Streamlit web interface
- `btc_pattern_finder.py` - Core pattern matching logic
- `data_provider.py` - Binance API data provider implementation
- `data_cache.py` - Local caching to minimize API calls  
- `visualization.py` - Pattern visualization utilities

## Future Enhancements

- Probabilistic price direction forecasting
- Real-time pattern detection
- Multiple asset comparison

## Requirements

- Python 3.8+
- STUMPY (for pattern matching)
- Pandas & NumPy (for data handling)
- Plotly (for visualization)
- Streamlit (for web interface)

## Notes

- The application uses Binance's public API to fetch real BTCUSDT spot market data
- For minute-level timeframes, data is available back to 2017
- No API key is required for accessing historical price data