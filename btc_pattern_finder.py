import numpy as np
import pandas as pd
import stumpy
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Literal, Optional
from feature_extraction import PriceFeatureExtractor

class BTCPatternFinder:
    """
    BTC Pattern Finder - Identifies similar patterns in Bitcoin price history
    Adapted from the original codebase's _find_similar_pattern function and
    CoinPriceChartFalshbackSearchUtility
    """
    
    # Adapted from interval_to_params in the original codebase
    # Modified allowed_day_difference to be very small - we only want to filter exact matches
    interval_to_params = {
        "1m": {"chart_length": 60, "fetch_days": 365, "allowed_day_difference": 0.01},
        "3m": {"chart_length": 60, "fetch_days": 365, "allowed_day_difference": 0.01},
        "5m": {"chart_length": 60, "fetch_days": 365, "allowed_day_difference": 0.01},
        "15m": {"chart_length": 48, "fetch_days": 365, "allowed_day_difference": 0.01},
        "30m": {"chart_length": 48, "fetch_days": 365, "allowed_day_difference": 0.01},
        "1h": {"chart_length": 24, "fetch_days": 365, "allowed_day_difference": 0.01},
        "4h": {"chart_length": 36, "fetch_days": 730, "allowed_day_difference": 0.01},
        "1d": {"chart_length": 30, "fetch_days": 2000, "allowed_day_difference": 0.01},
        "1w": {"chart_length": 114, "fetch_days": 4000, "allowed_day_difference": 0.01},
    }
    
    # Utility mapping to convert interval to points per day
    days_to_points_for_interval = {
        "1m": 24 * 60,  # 1440 points per day
        "3m": 24 * 20,  # 480 points per day
        "5m": 24 * 12,  # 288 points per day
        "15m": 24 * 4,  # 96 points per day
        "30m": 24 * 2,  # 48 points per day
        "1h": 24,       # 24 points per day
        "4h": 6,        # 6 points per day
        "1d": 1,        # 1 point per day
        "1w": 1/7,      # ~0.14 points per day
    }
    
    # Window sizes for market regime classification based on timeframe
    regime_window_sizes = {
        "1m": {"short_term": 120, "long_term": 720, "direction_change": 60},  # 2h, 12h, 1h
        "3m": {"short_term": 80, "long_term": 480, "direction_change": 40},   # 4h, 24h, 2h
        "5m": {"short_term": 72, "long_term": 288, "direction_change": 36},   # 6h, 24h, 3h
        "15m": {"short_term": 32, "long_term": 192, "direction_change": 16},  # 8h, 48h, 4h
        "30m": {"short_term": 16, "long_term": 96, "direction_change": 8},    # 8h, 48h, 4h
        "1h": {"short_term": 12, "long_term": 72, "direction_change": 6},     # 12h, 3d, 6h
        "2h": {"short_term": 12, "long_term": 72, "direction_change": 6},     # 24h, 6d, 12h
        "4h": {"short_term": 12, "long_term": 72, "direction_change": 6},     # 48h, 12d, 24h
        "6h": {"short_term": 8, "long_term": 56, "direction_change": 4},      # 48h, 14d, 24h
        "8h": {"short_term": 6, "long_term": 42, "direction_change": 3},      # 48h, 14d, 24h
        "12h": {"short_term": 4, "long_term": 28, "direction_change": 2},     # 48h, 14d, 24h
        "1d": {"short_term": 5, "long_term": 30, "direction_change": 3},      # 5d, 30d, 3d
        "3d": {"short_term": 4, "long_term": 20, "direction_change": 2},      # 12d, 60d, 6d
        "1w": {"short_term": 4, "long_term": 16, "direction_change": 2},      # 28d, 112d, 14d
    }
    
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.feature_extractor = None  # Will be initialized on demand
        
    def classify_market_regime(self, prices: pd.Series, interval: str, neutral_band: float = 0.2) -> int:
        """
        Classify market regime based on price action characteristics.
        
        Returns one of six regimes:
        1 = Bullish-Stable
        2 = Bullish-Volatile
        3 = Neutral
        4 = Bearish-Stable
        5 = Bearish-Volatile
        6 = Choppy (high direction changes)
        
        Args:
            prices: Series of price data (typically close prices)
            interval: Timeframe of the data (1m, 5m, 1h, etc.)
            neutral_band: Threshold around 0 for returns to be considered neutral
            
        Returns:
            int: Regime classification (1-6)
        """
        if len(prices) < 10:
            # Not enough data for reliable classification, default to neutral
            return 3
            
        # Get window sizes for this interval
        window_params = self.regime_window_sizes.get(interval, {"short_term": 5, "long_term": 30, "direction_change": 3})
        short_term = window_params["short_term"]
        long_term = window_params["long_term"]
        direction_window = window_params["direction_change"]
        
        # Ensure we have enough data
        if len(prices) < long_term + 10:
            # Not enough historical context, use what we have
            long_term = max(10, len(prices) // 2)
            short_term = max(5, long_term // 4)
            direction_window = max(3, short_term // 2)
            
        # Calculate returns for trend detection
        returns = prices.pct_change(short_term).dropna()
        
        # Calculate volatility
        volatility = returns.rolling(short_term).std().dropna()
        
        # Calculate direction changes for choppiness detection
        price_diff = prices.diff().fillna(0)
        direction = np.sign(price_diff)
        direction_changes = (direction.shift(1) != direction).astype(int)
        direction_change_freq = direction_changes.rolling(direction_window).sum() / direction_window
        
        # Get the most recent data point for each metric
        if len(returns) > 0 and len(volatility) > 0 and len(direction_change_freq) > 0:
            try:
                # Normalize metrics against their longer-term averages
                avg_returns = returns.rolling(long_term).mean()
                avg_volatility = volatility.rolling(long_term).mean()
                avg_direction_freq = direction_change_freq.rolling(long_term).mean()
                
                # Get the most recent values
                current_return = returns.iloc[-1]
                current_volatility = volatility.iloc[-1]
                current_direction_freq = direction_change_freq.iloc[-1]
                
                # Get reference values
                ref_return = avg_returns.iloc[-1] if not pd.isna(avg_returns.iloc[-1]) else 0
                ref_volatility = avg_volatility.iloc[-1] if not pd.isna(avg_volatility.iloc[-1]) else current_volatility
                ref_direction_freq = avg_direction_freq.iloc[-1] if not pd.isna(avg_direction_freq.iloc[-1]) else 0.5
                
                # Normalize relative to historical averages
                rel_return = current_return - ref_return
                rel_volatility = current_volatility / (ref_volatility if ref_volatility > 0 else 0.0001)
                rel_direction_freq = current_direction_freq / (ref_direction_freq if ref_direction_freq > 0 else 0.0001)
                
                # Check if market is choppy (high direction changes)
                is_choppy = rel_direction_freq > 1.2  # 20% more direction changes than normal
                
                if is_choppy:
                    # Choppy market regime (6)
                    return 6
                    
                # Classify based on trend and volatility
                if rel_return > neutral_band:
                    # Bullish
                    if rel_volatility <= 1.0:
                        return 1  # Bullish-Stable
                    else:
                        return 2  # Bullish-Volatile
                elif rel_return < -neutral_band:
                    # Bearish
                    if rel_volatility <= 1.0:
                        return 4  # Bearish-Stable
                    else:
                        return 5  # Bearish-Volatile
                else:
                    # Neutral
                    return 3
            except Exception as e:
                print(f"Error during regime classification: {str(e)}")
                return 3  # Default to neutral on error
        else:
            # Not enough data
            return 3  # Default to neutral
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get the name of a market regime by its ID."""
        regime_names = {
            1: "Bullish-Stable",
            2: "Bullish-Volatile",
            3: "Neutral",
            4: "Bearish-Stable",
            5: "Bearish-Volatile",
            6: "Choppy"
        }
        return regime_names.get(regime_id, "Unknown")
        
    def filter_matches_by_regime(
        self, 
        matches: List[Dict], 
        source_regime: int, 
        tolerance: int = 0,
        include_neutral: bool = True
    ) -> List[Dict]:
        """
        Filter pattern matches to include only those from similar market regimes.
        
        Args:
            matches: List of pattern match objects
            source_regime: Regime of the current pattern (1-6)
            tolerance: 0 means exact regime match only, 1 allows adjacent regimes
            include_neutral: Whether to always include neutral regime (3) matches
            
        Returns:
            List of matches from similar regimes
        """
        filtered_matches = []
        
        for match in matches:
            # Get the match's regime
            match_regime = match.get('regime', 3)  # Default to neutral if not available
            
            # Check if regimes match within tolerance
            if abs(match_regime - source_regime) <= tolerance:
                filtered_matches.append(match)
            # Always include neutral regime matches if requested
            elif include_neutral and match_regime == 3:
                filtered_matches.append(match)
                
        return filtered_matches
            
    def _find_similar_pattern(
        self,
        source_prices: Union[list, np.ndarray],
        target_prices: Union[list, np.ndarray],
        max_matches: int = 10,
    ) -> List[Tuple[float, int]]:
        """
        Find similar patterns using STUMPY's Matrix Profile algorithm
        Directly adapted from the original codebase's _find_similar_pattern function
        
        Args:
            source_prices: The price pattern to search for
            target_prices: The historical price data to search in
            max_matches: Maximum number of matches to return
            
        Returns:
            List of tuples containing (distance, index) of matches
        """
        source = np.array(source_prices)
        target = np.array(target_prices)
        
        pattern_length = len(source) - 1
        
        # Check for minimum pattern length - STUMPY requires at least 3 points for the query
        # (which means at least 4 candles for source_prices since we're using source[-pattern_length:])
        if pattern_length < 3:
            print(f"Error: Pattern length must be at least 4 candles (selected pattern has {pattern_length+1} candles)")
            return []
            
        if pattern_length >= len(target):
            raise ValueError("Pattern length must be shorter than both price series")
        
        query = source[-pattern_length:]
        
        try:
            # Using a more relaxed STUMPY configuration to get more matches
            stumpy.config.STUMPY_EXCL_ZONE_DENOM = np.inf
            
            # Request exactly the number of matches specified by the user's slider
            # But multiply by 10 internally to ensure we have enough before filtering
            internal_max = max_matches * 10  # Get 10x the requested amount to ensure enough after filtering
            matches = stumpy.match(
                query,
                target,
                max_distance=np.inf,  # No maximum distance restriction
                max_matches=internal_max,  # Get more than requested to ensure enough after filtering
            )
            
            # Restore default exclusion zone
            stumpy.config.STUMPY_EXCL_ZONE_DENOM = 4
            
            # Convert to list of tuples if it's a numpy array
            # Also filter out any potential NaN distances
            if isinstance(matches, np.ndarray):
                return [(float(dist), int(idx)) for dist, idx in matches if not np.isnan(dist)]
            else:
                return matches
        
        except Exception as e:
            print(f"Error during pattern matching: {str(e)}")
            return []
            
    async def find_similar_patterns(
        self,
        symbol: str = "BTC",  # Added symbol parameter with BTC default
        interval: str = "1d",
        chart_length: int = None,
        start_time: str = None,
        end_time: str = None,
        max_matches: int = 10,
        following_points: int = 20,
        search_start_time: str = None,
        search_end_time: str = None,
        source_idx_range: Tuple[int, int] = None,  # Add parameter for source index range (DEPRECATED)
        source_pattern: List[Dict] = None,  # NEW: Explicit source pattern data
        use_regime_filter: bool = False,  # Whether to filter by market regime
        regime_tolerance: int = 0,  # How strict to be with regime matching (0=exact, 1=adjacent)
        include_neutral: bool = True,  # Whether to include neutral regime matches
    ) -> Dict:
        """
        Find historical patterns similar to a specified time range
        Adapted from the CoinPriceChartFalshbackSearchUtility.arun method
        
        Args:
            symbol: Asset symbol (e.g., "BTC", "ETH") to analyze
            interval: Timeframe for pattern matching (1m, 5m, 1h, 1d, etc.)
            chart_length: Number of candles to use for the pattern
            start_time: ISO format start time for pattern (if None, uses recent data)
            end_time: ISO format end time for pattern (if None, uses recent data)
            max_matches: Maximum number of matches to return
            following_points: Number of data points to include after each match
            search_start_time: Optional ISO format start time to limit the search range
            search_end_time: Optional ISO format end time to limit the search range
            
        Returns:
            Dictionary with pattern matches and metadata
        """
        params = self.interval_to_params[interval]
        
        if chart_length is None:
            chart_length = params["chart_length"]
            
        # NEW: Handle the case when source_pattern is directly provided
        if source_pattern is not None:
            # Convert the provided pattern directly to a DataFrame
            source_ohlcv = pd.DataFrame(source_pattern)
            
            # We still need start/end time for debug print
            if 'timestamp' in source_ohlcv.columns:
                # Get timestamps from the data
                first_ts = source_ohlcv['timestamp'].iloc[0]
                last_ts = source_ohlcv['timestamp'].iloc[-1]
                
                # Convert to datetime for display
                start_time = datetime.fromtimestamp(first_ts / 1000)
                end_time = datetime.fromtimestamp(last_ts / 1000)
            else:
                # Default values if timestamps not available
                start_time = datetime.now()
                end_time = datetime.now()
        else:
            # Original logic for timestamp-based pattern extraction
            current_time = datetime.now()
            
            if start_time is None and end_time is None:
                # Default to recent data
                end_time = current_time
                days_back = chart_length / self.days_to_points_for_interval[interval]
                start_time = end_time - timedelta(days=days_back)
            elif start_time is not None and end_time is None:
                # Use specified start and calculate end
                start_time = datetime.fromisoformat(start_time)
                days_forward = chart_length / self.days_to_points_for_interval[interval]
                end_time = start_time + timedelta(days=days_forward)
            elif start_time is None and end_time is not None:
                # Use specified end and calculate start
                end_time = datetime.fromisoformat(end_time)
                days_back = chart_length / self.days_to_points_for_interval[interval]
                start_time = end_time - timedelta(days=days_back)
            else:
                # Both specified
                start_time = datetime.fromisoformat(start_time)
                end_time = datetime.fromisoformat(end_time)
                
            # Ensure end_time doesn't exceed current time
            if end_time > current_time:
                end_time = current_time
                
            # Fetch source pattern (current or specified time range)
            source_ohlcv = await self.data_provider.get_historical_ohlcv(
                symbol=symbol,  # Use the provided symbol
                interval=interval,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
        
        # Source pattern validation
        if len(source_ohlcv) == 0:
            return {"error": "No data found for source pattern"}
            
        # Check if source pattern is too short (needs at least 4 candles)
        if len(source_ohlcv) < 4:
            return {"error": f"Pattern is too short. Please select at least 4 candles. (Selected: {len(source_ohlcv)} candles)"}
            
        # Fetch historical data for matching (extensive history)
        fetch_days = params["fetch_days"]
        
        # Use search range parameters if provided, otherwise use default range
        if search_start_time is not None:
            historical_start = datetime.fromisoformat(search_start_time)
        else:
            historical_start = current_time - timedelta(days=fetch_days)
        
        if search_end_time is not None:
            historical_end = datetime.fromisoformat(search_end_time)
        else:
            historical_end = current_time
        
        historical_ohlcv = await self.data_provider.get_historical_ohlcv(
            symbol=symbol,  # Use the provided symbol
            interval=interval,
            start_time=historical_start.isoformat(),
            end_time=historical_end.isoformat()
        )
        
        if len(historical_ohlcv) == 0:
            return {"error": "No historical data found"}
            
        # Ensure we have at least 4 candles in the source pattern
        if len(source_ohlcv) < 4:
            return {"error": f"Pattern is too short. Please select at least 4 candles. (Selected: {len(source_ohlcv)} candles)"}
        
        # Find matches using the adapted function from the original codebase
        matches = self._find_similar_pattern(
            source_ohlcv["close"].values,
            historical_ohlcv["close"].values,
            max_matches=max_matches
        )
        
        if matches is None or len(matches) == 0:
            return {"error": "No matches found"}
            
        # Process matches
        match_results = []
        match_dates = []
        
        historical_dates = pd.to_datetime(historical_ohlcv["timestamp"], unit='ms')
        
        # Determine the source pattern's absolute timestamp range
        # This is the range we want to exclude from results
        source_timestamps = pd.to_datetime(source_ohlcv["timestamp"], unit='ms')
        source_start_timestamp = source_timestamps.min()
        source_end_timestamp = source_timestamps.max()
        
        # Use a minimal buffer - just enough to ensure exact timestamp matching works
        buffer = pd.Timedelta(seconds=1)
        
        # Apply buffer to exclusion zone
        exclude_start = source_start_timestamp - buffer
        exclude_end = source_end_timestamp + buffer
        
        # Process each match
        for match_dist, match_idx in matches:
            if match_idx >= len(historical_ohlcv):
                continue  # Skip invalid indices
                
            # Get match timestamps - convert full pattern to datetime
            match_times = pd.to_datetime(historical_ohlcv.iloc[match_idx:match_idx+len(source_ohlcv)]["timestamp"], unit='ms')
            
            # Only use valid timestamps (in case we're at the end of data)
            if not match_times.empty:
                match_start_time = match_times.min()
                match_end_time = match_times.max()
                match_datetime = match_start_time  # For compatibility
                
                # Check for overlap with the excluded zone
                # A match overlaps if ANY part of it is within the exclude zone
                no_overlap = (match_end_time < exclude_start) or (match_start_time > exclude_end)
                
                # Additional check to prevent similar shifted matches
                too_close_to_existing = False
                if no_overlap:
                    # Check if this match is too close to any existing match
                    for existing_date in match_dates:
                        # Calculate time difference in seconds
                        time_diff = abs((match_datetime - existing_date).total_seconds())
                        
                        # Convert interval to seconds for threshold
                        if interval == "1d":
                            threshold = 60 * 60 * 24 * 3  # 3 days
                        elif interval == "1h":
                            threshold = 60 * 60 * 8       # 8 hours (increased from 5)
                        elif interval == "4h":
                            threshold = 60 * 60 * 12      # 12 hours (increased from 10)
                        elif interval == "30m":
                            threshold = 60 * 30 * 12      # 6 hours (increased from 2.5 hours)
                        elif interval == "15m":
                            threshold = 60 * 15 * 16      # 4 hours (increased from ~2 hours)
                        elif interval == "5m":
                            threshold = 60 * 5 * 36       # 3 hours (increased from ~1.25 hours)
                        elif interval == "3m":
                            threshold = 60 * 3 * 60       # 3 hours
                        elif interval == "1m":
                            threshold = 60 * 1 * 180      # 3 hours
                        else:
                            threshold = 60 * 60 * 3       # 3 hours default (increased from 2)
                            
                        if time_diff < threshold:
                            too_close_to_existing = True
                            break
                
                # More permissive filtering - just exclude exact pattern and duplicates
                if no_overlap and not too_close_to_existing and match_dist > 0.01:  # Multiple filters
                    match_dates.append(match_datetime)
                    
                    # Get data points after match for forward analysis
                    # Make sure we get EXACTLY chart_length + following_points candles
                    # chart_length is the pattern part, following_points is the future part
                    target_length = chart_length + following_points
                    available_length = min(len(historical_ohlcv) - match_idx, target_length)
                    
                    # If we have enough data, use the requested amount
                    # Otherwise, use what's available but print a warning
                    if available_length < target_length:
                        print(f"Warning: Not enough data for match at {match_idx}. " 
                              f"Requested {target_length} candles but only {available_length} available.")
                        
                    match_data = historical_ohlcv.iloc[match_idx:match_idx + available_length]
                
                    # Calculate start and end times
                    match_start = match_data.iloc[0]["timestamp"]
                    match_end = match_data.iloc[-1]["timestamp"]
                    
                    # Classify market regime for this match
                    match_regime = self.classify_market_regime(match_data["close"], interval)
                    
                    match_results.append({
                        "distance": float(match_dist),
                        "start_time": datetime.fromtimestamp(match_start / 1000).isoformat(),
                        "end_time": datetime.fromtimestamp(match_end / 1000).isoformat(),
                        "timestamp": int(match_start),
                        "pattern_data": match_data[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records"),
                        "label": f"{symbol} from {match_datetime.strftime('%Y-%m-%d %H:%M')}",
                        "regime": match_regime,
                        "regime_name": self.get_regime_name(match_regime)
                    })
                
        # Sort by distance (similarity)
        sorted_results = sorted(match_results, key=lambda x: x["distance"])
        
        # Classify market regime for source pattern
        source_regime = self.classify_market_regime(source_ohlcv["close"], interval)
        source_regime_name = self.get_regime_name(source_regime)
        
        # Apply regime filtering if requested
        regime_filtered_results = sorted_results
        if use_regime_filter:
            regime_filtered_results = self.filter_matches_by_regime(
                sorted_results, 
                source_regime, 
                tolerance=regime_tolerance,
                include_neutral=include_neutral
            )
            print(f"Regime filtering: {len(sorted_results)} matches → {len(regime_filtered_results)} matches")
        
        # Apply a limit to the results based on the original max_matches request
        final_results = regime_filtered_results[:max_matches] if len(regime_filtered_results) > max_matches else regime_filtered_results
        
        # Store filtered scores (after time proximity filtering) for distribution visualization
        # This will exclude near-duplicates and only show meaningful differences
        filtered_raw_scores = [match['distance'] for match in sorted_results]
        
        return {
            "type": "flashback_pattern",
            "symbol": symbol,  # Include the symbol in the response
            "interval": interval,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "following_points_number": following_points,
            "total_matches": len(final_results),
            "flashback_patterns": final_results,
            "filtered_matches_scores": filtered_raw_scores,  # Include only time-filtered scores for histogram
            "source_regime": source_regime,
            "source_regime_name": source_regime_name,
            "debug_info": {
                "requested_matches": max_matches,
                "stumpy_matches_found": len(matches) if matches else 0,
                "unique_matches": len(sorted_results),  # Matches after time-proximity filtering
                "regime_filtered_matches": len(regime_filtered_results) if use_regime_filter else None,
                "final_matches": len(final_results)
            }
        }
        
    def _find_similar_pattern_feature_based(
        self,
        source_ohlcv: pd.DataFrame,
        historical_ohlcv: pd.DataFrame,
        window_size: Optional[int] = None,
        stride: int = 1,
        n_components: int = 2,
        max_matches: int = 10,
    ) -> Tuple[List[Tuple[float, int]], Dict]:
        """
        Find similar patterns using PCA feature extraction approach

        Args:
            source_ohlcv: DataFrame with the source pattern OHLCV data
            historical_ohlcv: DataFrame with historical OHLCV data to search in
            window_size: Size of the sliding window (default: len(source_ohlcv))
            stride: Steps between windows
            n_components: Number of components to use
            max_matches: Maximum number of matches to return

        Returns:
            Tuple of (matches, visualization_data)
            - matches: List of tuples with (distance, index)
            - visualization_data: Dict with data for visualizing feature space
        """
        if window_size is None:
            window_size = len(source_ohlcv)
            
        # Initialize the feature extractor if needed
        if self.feature_extractor is None or self.feature_extractor.window_size != window_size:
            self.feature_extractor = PriceFeatureExtractor(
                window_size=window_size,
                n_components=n_components
            )
            
        # Find similar patterns using our feature extractor
        matches, vis_data = self.feature_extractor.find_similar_patterns(
            source_ohlc=source_ohlcv,
            historical_ohlc=historical_ohlcv,
            window_size=window_size,
            stride=stride,
            top_n=max_matches * 10  # Get 10x requested to ensure enough after filtering
        )
        
        return matches, vis_data
        
    async def find_similar_patterns_feature_based(
        self,
        symbol: str = "BTC",
        interval: str = "1d",
        chart_length: int = None,
        start_time: str = None,
        end_time: str = None,
        max_matches: int = 10,
        following_points: int = 20,
        search_start_time: str = None,
        search_end_time: str = None,
        source_idx_range: Tuple[int, int] = None,
        source_pattern: List[Dict] = None,
        n_components: int = 2,
        use_regime_filter: bool = False,  # Whether to filter by market regime
        regime_tolerance: int = 0,  # How strict to be with regime matching (0=exact, 1=adjacent)
        include_neutral: bool = True,  # Whether to include neutral regime matches
    ) -> Dict:
        """
        Find historical patterns similar to a specified time range using feature extraction
        
        Args:
            symbol: Asset symbol (e.g., "BTC", "ETH") to analyze
            interval: Timeframe for pattern matching (1m, 5m, 1h, 1d, etc.)
            chart_length: Number of candles to use for the pattern
            start_time: ISO format start time for pattern (if None, uses recent data)
            end_time: ISO format end time for pattern (if None, uses recent data)
            max_matches: Maximum number of matches to return
            following_points: Number of data points to include after each match
            search_start_time: Optional ISO format start time to limit the search range
            search_end_time: Optional ISO format end time to limit the search range
            source_idx_range: Optional tuple with start and end indices for source pattern
            source_pattern: Optional explicit source pattern data
            n_components: Number of components to use
            
        Returns:
            Dictionary with pattern matches and metadata
        """
        # Most logic is identical to find_similar_patterns, with only pattern matching algorithm changed
        params = self.interval_to_params[interval]
        
        if chart_length is None:
            chart_length = params["chart_length"]
            
        # Handle the case when source_pattern is directly provided
        if source_pattern is not None:
            # Convert the provided pattern directly to a DataFrame
            source_ohlcv = pd.DataFrame(source_pattern)
            
            # We still need start/end time for debug print
            if 'timestamp' in source_ohlcv.columns:
                # Get timestamps from the data
                first_ts = source_ohlcv['timestamp'].iloc[0]
                last_ts = source_ohlcv['timestamp'].iloc[-1]
                
                # Convert to datetime for display
                start_time = datetime.fromtimestamp(first_ts / 1000)
                end_time = datetime.fromtimestamp(last_ts / 1000)
            else:
                # Default values if timestamps not available
                start_time = datetime.now()
                end_time = datetime.now()
        else:
            # Original logic for timestamp-based pattern extraction
            current_time = datetime.now()
            
            if start_time is None and end_time is None:
                # Default to recent data
                end_time = current_time
                days_back = chart_length / self.days_to_points_for_interval[interval]
                start_time = end_time - timedelta(days=days_back)
            elif start_time is not None and end_time is None:
                # Use specified start and calculate end
                start_time = datetime.fromisoformat(start_time)
                days_forward = chart_length / self.days_to_points_for_interval[interval]
                end_time = start_time + timedelta(days=days_forward)
            elif start_time is None and end_time is not None:
                # Use specified end and calculate start
                end_time = datetime.fromisoformat(end_time)
                days_back = chart_length / self.days_to_points_for_interval[interval]
                start_time = end_time - timedelta(days=days_back)
            else:
                # Both specified
                start_time = datetime.fromisoformat(start_time)
                end_time = datetime.fromisoformat(end_time)
                
            # Ensure end_time doesn't exceed current time
            if end_time > current_time:
                end_time = current_time
                
            # Fetch source pattern (current or specified time range)
            source_ohlcv = await self.data_provider.get_historical_ohlcv(
                symbol=symbol,
                interval=interval,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
        
        # Source pattern validation
        if len(source_ohlcv) == 0:
            return {"error": "No data found for source pattern"}
            
        # Check if source pattern is too short
        if len(source_ohlcv) < 4:
            return {"error": f"Pattern is too short. Please select at least 4 candles. (Selected: {len(source_ohlcv)} candles)"}
            
        # Fetch historical data for matching (extensive history)
        fetch_days = params["fetch_days"]
        
        # Use search range parameters if provided, otherwise use default range
        if search_start_time is not None:
            historical_start = datetime.fromisoformat(search_start_time)
        else:
            historical_start = datetime.now() - timedelta(days=fetch_days)
        
        if search_end_time is not None:
            historical_end = datetime.fromisoformat(search_end_time)
        else:
            historical_end = datetime.now()
        
        historical_ohlcv = await self.data_provider.get_historical_ohlcv(
            symbol=symbol,
            interval=interval,
            start_time=historical_start.isoformat(),
            end_time=historical_end.isoformat()
        )
        
        if len(historical_ohlcv) == 0:
            return {"error": "No historical data found"}
            
        # Performance optimization: Use stride of max(1, len(source_ohlcv)//5) for faster processing
        # This means we'll check every 5th candle, which dramatically speeds up processing
        # but still finds most of the important patterns
        stride = max(1, len(source_ohlcv)//5)
        print(f"Using stride of {stride} for feature extraction (sampling every {stride}th candle)")
        
        # Use feature-based pattern matching
        matches, vis_data = self._find_similar_pattern_feature_based(
            source_ohlcv=source_ohlcv,
            historical_ohlcv=historical_ohlcv,
            window_size=len(source_ohlcv),
            stride=stride,  # Use stride for faster processing
            n_components=n_components,
            max_matches=max_matches
        )
        
        if matches is None or len(matches) == 0:
            return {"error": "No matches found"}
            
        # Process matches - same logic as find_similar_patterns
        match_results = []
        match_dates = []
        match_indices = []  # Store matched indices for visualization
        
        historical_dates = pd.to_datetime(historical_ohlcv["timestamp"], unit='ms')
        
        # Determine the source pattern's absolute timestamp range to exclude
        source_timestamps = pd.to_datetime(source_ohlcv["timestamp"], unit='ms')
        source_start_timestamp = source_timestamps.min()
        source_end_timestamp = source_timestamps.max()
        
        # Use a minimal buffer
        buffer = pd.Timedelta(seconds=1)
        exclude_start = source_start_timestamp - buffer
        exclude_end = source_end_timestamp + buffer
        
        # Process each match
        window_size = len(source_ohlcv)
        
        for match_dist, match_idx in matches:
            if match_idx >= len(historical_ohlcv):
                continue  # Skip invalid indices
                
            # Get match timestamps
            match_times = pd.to_datetime(historical_ohlcv.iloc[match_idx:match_idx+window_size]["timestamp"], unit='ms')
            
            if not match_times.empty:
                match_start_time = match_times.min()
                match_end_time = match_times.max()
                match_datetime = match_start_time
                
                # Check for overlap with the excluded zone
                no_overlap = (match_end_time < exclude_start) or (match_start_time > exclude_end)
                
                # Check for proximity to existing matches
                too_close_to_existing = False
                if no_overlap:
                    for existing_date in match_dates:
                        time_diff = abs((match_datetime - existing_date).total_seconds())
                        
                        # Convert interval to seconds for threshold
                        if interval == "1d":
                            threshold = 60 * 60 * 24 * 3  # 3 days
                        elif interval == "1h":
                            threshold = 60 * 60 * 8       # 8 hours (increased from 5)
                        elif interval == "4h":
                            threshold = 60 * 60 * 12      # 12 hours (increased from 10)
                        elif interval == "30m":
                            threshold = 60 * 30 * 12      # 6 hours (increased from 2.5 hours)
                        elif interval == "15m":
                            threshold = 60 * 15 * 16      # 4 hours (increased from ~2 hours)
                        elif interval == "5m":
                            threshold = 60 * 5 * 36       # 3 hours (increased from ~1.25 hours)
                        elif interval == "3m":
                            threshold = 60 * 3 * 60       # 3 hours
                        elif interval == "1m":
                            threshold = 60 * 1 * 180      # 3 hours
                        else:
                            threshold = 60 * 60 * 3       # 3 hours default (increased from 2)
                            
                        if time_diff < threshold:
                            too_close_to_existing = True
                            break
                
                # Apply filtering
                if no_overlap and not too_close_to_existing:
                    match_dates.append(match_datetime)
                    match_indices.append(match_idx)
                    
                    # Get data points after match for forward analysis
                    target_length = window_size + following_points
                    available_length = min(len(historical_ohlcv) - match_idx, target_length)
                    
                    if available_length < target_length:
                        print(f"Warning: Not enough data for match at {match_idx}. " 
                              f"Requested {target_length} candles but only {available_length} available.")
                        
                    match_data = historical_ohlcv.iloc[match_idx:match_idx + available_length]
                
                    # Calculate start and end times
                    match_start = match_data.iloc[0]["timestamp"]
                    match_end = match_data.iloc[-1]["timestamp"]
                    
                    # Classify market regime for this match
                    match_regime = self.classify_market_regime(match_data["close"], interval)
                    
                    match_results.append({
                        "distance": float(match_dist),
                        "start_time": datetime.fromtimestamp(match_start / 1000).isoformat(),
                        "end_time": datetime.fromtimestamp(match_end / 1000).isoformat(),
                        "timestamp": int(match_start),
                        "pattern_data": match_data[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records"),
                        "label": f"{symbol} from {match_datetime.strftime('%Y-%m-%d %H:%M')}",
                        "regime": match_regime,
                        "regime_name": self.get_regime_name(match_regime)
                    })
        
        # Sort by distance
        sorted_results = sorted(match_results, key=lambda x: x["distance"])
        
        # Classify market regime for source pattern
        source_regime = self.classify_market_regime(source_ohlcv["close"], interval)
        source_regime_name = self.get_regime_name(source_regime)
        
        # Apply regime filtering if requested
        regime_filtered_results = sorted_results
        if use_regime_filter:
            regime_filtered_results = self.filter_matches_by_regime(
                sorted_results, 
                source_regime, 
                tolerance=regime_tolerance,
                include_neutral=include_neutral
            )
            print(f"Regime filtering: {len(sorted_results)} matches → {len(regime_filtered_results)} matches")
            
            # Update match indices for visualization if using regime filtering
            match_indices = [match_indices[sorted_results.index(match)] for match in regime_filtered_results if sorted_results.index(match) < len(match_indices)]
        
        # Apply max_matches limit
        final_results = regime_filtered_results[:max_matches] if len(regime_filtered_results) > max_matches else regime_filtered_results
        
        # Store filtered scores for distribution visualization
        filtered_raw_scores = [match['distance'] for match in sorted_results]
        
        return {
            "type": "feature_pattern",
            "symbol": symbol,
            "interval": interval,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "following_points_number": following_points,
            "total_matches": len(final_results),
            "method": "pca",
            "n_components": n_components,
            "flashback_patterns": final_results,
            "filtered_matches_scores": filtered_raw_scores,
            "vis_data": vis_data,
            "match_indices": match_indices,
            "source_regime": source_regime,
            "source_regime_name": source_regime_name,
            "debug_info": {
                "requested_matches": max_matches,
                "feature_matches_found": len(matches) if matches else 0,
                "unique_matches": len(sorted_results),
                "regime_filtered_matches": len(regime_filtered_results) if use_regime_filter else None,
                "final_matches": len(final_results)
            }
        }