import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

class DataCache:
    """
    Cache for historical OHLCV data to minimize API calls
    """
    
    def __init__(self, db_path: str = "btc_pattern_cache.db"):
        self.db_path = db_path
        self._create_tables()
        
    def _create_tables(self):
        """Create the cache tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create OHLCV cache table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_cache (
            symbol TEXT,
            interval TEXT,
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            cached_at INTEGER,
            PRIMARY KEY (symbol, interval, timestamp)
        )
        """)
        
        conn.commit()
        conn.close()
        
    def get_cached_data(
        self,
        symbol: str,
        interval: str,
        start_timestamp: int,
        end_timestamp: int
    ) -> pd.DataFrame:
        """
        Retrieve cached OHLCV data for a symbol and interval
        
        Args:
            symbol: Asset symbol (e.g., "BTC")
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds
            
        Returns:
            DataFrame with cached OHLCV data, or empty DataFrame if not found
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_cache
        WHERE symbol = ? AND interval = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, interval, start_timestamp, end_timestamp)
        )
        
        conn.close()
        
        return df
        
    def save_data(self, symbol: str, interval: str, data: pd.DataFrame):
        """
        Save OHLCV data to cache
        
        Args:
            symbol: Asset symbol (e.g., "BTC")
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            data: DataFrame with OHLCV data
        """
        if data.empty:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = int(datetime.now().timestamp())
        
        # Prepare data for insertion
        for _, row in data.iterrows():
            cursor.execute(
                """
                INSERT OR REPLACE INTO ohlcv_cache
                (symbol, interval, timestamp, open, high, low, close, volume, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    interval,
                    int(row["timestamp"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    current_time
                )
            )
            
        conn.commit()
        conn.close()
        
    def get_missing_ranges(
        self,
        symbol: str,
        interval: str,
        start_timestamp: int,
        end_timestamp: int,
        max_age_hours: int = 24
    ) -> List[Dict[str, int]]:
        """
        Identify missing or stale data ranges that need to be fetched
        
        Args:
            symbol: Asset symbol (e.g., "BTC")
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds
            max_age_hours: Maximum age of cached data in hours
            
        Returns:
            List of dictionaries with start and end timestamps for missing ranges
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate stale timestamp threshold
        stale_threshold = int((datetime.now() - timedelta(hours=max_age_hours)).timestamp())
        
        # Get all cached timestamps in the range
        cursor.execute(
            """
            SELECT timestamp, cached_at
            FROM ohlcv_cache
            WHERE symbol = ? AND interval = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            """,
            (symbol, interval, start_timestamp, end_timestamp)
        )
        
        cached_data = cursor.fetchall()
        conn.close()
        
        # Convert to set for faster lookups
        fresh_timestamps = {
            ts for ts, cached_at in cached_data
            if cached_at >= stale_threshold
        }
        
        # Determine interval in milliseconds
        interval_map = {
            "1m": 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
        }
        
        interval_ms = interval_map.get(interval, 24 * 60 * 60 * 1000)
        
        # Generate all expected timestamps in the range
        expected_timestamps = set()
        current = start_timestamp
        while current <= end_timestamp:
            expected_timestamps.add(current)
            current += interval_ms
            
        # Find missing timestamps
        missing_timestamps = expected_timestamps - fresh_timestamps
        
        if not missing_timestamps:
            return []
            
        # Group missing timestamps into continuous ranges
        missing_timestamps = sorted(missing_timestamps)
        missing_ranges = []
        
        range_start = missing_timestamps[0]
        prev_ts = missing_timestamps[0]
        
        for ts in missing_timestamps[1:]:
            if ts - prev_ts > interval_ms:
                missing_ranges.append({
                    "start": range_start,
                    "end": prev_ts
                })
                range_start = ts
            prev_ts = ts
            
        # Add the last range
        missing_ranges.append({
            "start": range_start,
            "end": prev_ts
        })
        
        return missing_ranges