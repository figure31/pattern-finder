import aiohttp
import pandas as pd
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pytz

class BaseDataProvider:
    """
    Base class for data providers
    Adapted from the original BaseOHLCVProvider
    """
    
    async def get_historical_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Asset symbol (e.g., "BTC")
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Optional ISO format start time
            end_time: Optional ISO format end time
            days: Number of days of historical data (default: 365)
            
        Returns:
            pandas.DataFrame with OHLCV data
        """
        raise NotImplementedError("Subclasses must implement this method")


class BinanceDataProvider(BaseDataProvider):
    """
    Data provider using Binance API for BTCUSDT spot market data
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.us/api/v3"
        self._max_retries = 3
        self._request_delay = 0.1  # 100ms between requests
        
    async def get_historical_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance
        
        Args:
            symbol: Asset symbol (e.g., "BTC", "ETH")
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Optional ISO format start time
            end_time: Optional ISO format end time
            days: Number of days of historical data (default: 365)
            
        Returns:
            pandas.DataFrame with OHLCV data
        """
        # Format symbol for Binance (add USDT suffix)
        if symbol.upper() == "ETH":
            formatted_symbol = "ETHUSDT"
        else:
            # Default to BTC if not explicitly ETH
            formatted_symbol = "BTCUSDT"
        
        # Determine time range
        if end_time:
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        else:
            end = datetime.now()
            
        if start_time:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        else:
            start = end - timedelta(days=days)
        
        # Convert to milliseconds for Binance API
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        
        # Calculate time spans for chunking requests
        time_span_ms = end_ms - start_ms
        
        # Binance API URL for klines (candlestick data)
        url = f"{self.base_url}/klines"
        
        # For long time periods, we need to make multiple requests due to the 1000 candle limit
        chunk_results = await self._fetch_data_in_chunks(url, formatted_symbol, interval, start_ms, end_ms)
        
        if not chunk_results:
            # If API fails, just return empty dataframe with proper error message
            print("Error: Failed to fetch data from Binance API")
            return pd.DataFrame()
            
        # Combine all chunks
        all_klines = []
        for chunk in chunk_results:
            all_klines.extend(chunk)
            
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades_count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignored"
        ])
        
        # Convert types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
            
        # Ensure proper sorting and reset index
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Return only the columns we need
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    async def _fetch_data_in_chunks(
        self, 
        url: str, 
        symbol: str, 
        interval: str, 
        start_ms: int, 
        end_ms: int,
        chunk_size: int = 1000
    ) -> List[List[List]]:
        """
        Fetch data in chunks to handle the 1000 candle limit of Binance API
        
        Args:
            url: API endpoint URL
            symbol: Trading pair symbol
            interval: Candle interval
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            chunk_size: Maximum number of candles per request
            
        Returns:
            List of kline data chunks
        """
        # Calculate the approximate size of each chunk based on the interval
        interval_ms = self._interval_to_milliseconds(interval)
        if interval_ms == 0:
            return []
            
        # Calculate the number of candles needed
        total_candles = (end_ms - start_ms) // interval_ms
        
        # Calculate chunk points (start timestamps for each chunk)
        chunk_points = []
        current = start_ms
        
        while current < end_ms:
            chunk_points.append(current)
            # Each chunk will fetch up to 1000 candles or until end_time
            current += interval_ms * chunk_size
            
        # Add the end time to ensure we get the complete range
        if chunk_points[-1] + (interval_ms * chunk_size) < end_ms:
            chunk_points.append(end_ms - (interval_ms * chunk_size))
            
        # Prepare async tasks to fetch each chunk
        try:
            # Use a longer timeout for cloud deployment
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = []
                
                for i, chunk_start in enumerate(chunk_points):
                    chunk_end = min(chunk_start + (interval_ms * chunk_size), end_ms)
                    
                    # Prepare the parameters for this chunk
                    params = {
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": chunk_start,
                        "endTime": chunk_end,
                        "limit": chunk_size
                    }
                    
                    # Create task for this chunk
                    task = self._fetch_chunk(session, url, params, f"Chunk {i+1}/{len(chunk_points)}")
                    tasks.append(task)
                    
                # Run all tasks concurrently with a small delay between each to avoid rate limits
                results = []
                for i, task in enumerate(tasks):
                    # Add a small delay between requests to avoid rate limits
                    if i > 0:
                        await asyncio.sleep(self._request_delay)
                    
                    try:    
                        chunk_result = await task
                        if chunk_result:
                            results.append(chunk_result)
                    except Exception as e:
                        print(f"Task {i} failed with error: {str(e)}")
                        
                return results
        except Exception as e:
            print(f"Fatal error in fetch_data_in_chunks: {str(e)}")
            # In case of catastrophic failure, return empty results
            return []
            
    async def _fetch_chunk(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        params: Dict,
        chunk_id: str
    ) -> Optional[List[List]]:
        """
        Fetch a single chunk of data from Binance API
        
        Args:
            session: aiohttp ClientSession
            url: API endpoint URL
            params: Request parameters
            chunk_id: Identifier for this chunk (for logging)
            
        Returns:
            List of kline data or None if request failed
        """
        # Add retries for reliability
        for attempt in range(self._max_retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        try:
                            error_text = await response.text()
                            print(f"Error fetching {chunk_id}: {response.status} - {error_text}")
                        except:
                            print(f"Error fetching {chunk_id}: Status {response.status}, could not read response text")
                        
                        # If we hit a rate limit, wait and try again
                        if response.status == 429:
                            retry_after = int(response.headers.get('Retry-After', 1))
                            print(f"Rate limited, retrying after {retry_after} seconds")
                            await asyncio.sleep(retry_after)
                            continue
                        elif response.status in [500, 502, 503, 504]:
                            # Server error, wait longer
                            wait_time = 2 ** attempt  # Exponential backoff
                            print(f"Server error, retrying after {wait_time} seconds")
                            await asyncio.sleep(wait_time)
                            continue
                            
                        return None
                    
                    try:    
                        klines = await response.json()
                    except Exception as e:
                        print(f"Failed to parse JSON response for {chunk_id}: {str(e)}")
                        # Try to get the raw text
                        try:
                            text = await response.text()
                            print(f"Raw response: {text[:200]}...")  # First 200 chars
                        except:
                            print("Could not read response text")
                        
                        # Wait and retry
                        await asyncio.sleep(1)
                        continue
                    
                    if not klines or not isinstance(klines, list):
                        print(f"No valid data returned for {chunk_id}")
                        return None
                        
                    print(f"Successfully fetched {len(klines)} candles for {chunk_id}")
                    return klines
                    
            except aiohttp.ClientConnectorError as e:
                print(f"Connection error for {chunk_id} (attempt {attempt+1}/{self._max_retries}): {str(e)}")
                await asyncio.sleep(2)  # Longer wait for connection issues
            except asyncio.TimeoutError:
                print(f"Timeout error for {chunk_id} (attempt {attempt+1}/{self._max_retries})")
                await asyncio.sleep(2)  # Longer wait for timeouts
            except Exception as e:
                print(f"Exception during {chunk_id} fetch (attempt {attempt+1}/{self._max_retries}): {str(e)}")
                await asyncio.sleep(1)  # Wait before retry
                
        # All retries failed
        print(f"All retries failed for {chunk_id}")
        return None
        
    def _interval_to_milliseconds(self, interval: str) -> int:
        """
        Convert interval string to milliseconds
        
        Args:
            interval: Interval string (e.g. "1m", "1h", "1d")
            
        Returns:
            Interval in milliseconds
        """
        # Parse the interval string
        unit = interval[-1]
        value = int(interval[:-1])
        
        # Convert to milliseconds
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60 * 1000
        else:
            return 0  # Invalid interval
    
    # Simulated data function removed as it's not needed for production use


class CompositeBTCDataProvider(BaseDataProvider):
    """
    Composite implementation for BTC data
    Uses multiple providers with failover capability
    Adapted from the original CompositeCoinDataProvider
    """
    
    def __init__(self, data_providers: List[BaseDataProvider] = None):
        """
        Initialize with a list of data providers, or create a default Binance provider
        """
        if data_providers is None:
            # Default to using Binance provider
            self.data_providers = [BinanceDataProvider()]
        else:
            self.data_providers = data_providers
        
    async def get_historical_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data using multiple providers with failover
        Tries each provider in order until successful
        
        Args:
            symbol: Asset symbol (e.g., "BTC")
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Optional ISO format start time
            end_time: Optional ISO format end time
            days: Number of days of historical data (default: 365)
            
        Returns:
            pandas.DataFrame with OHLCV data
        """
        for provider in self.data_providers:
            try:
                data = await provider.get_historical_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    days=days
                )
                
                if not data.empty:
                    return data
            except Exception as e:
                print(f"Provider {provider.__class__.__name__} failed: {str(e)}")
                continue
                
        # If all providers fail, return empty DataFrame
        return pd.DataFrame()
