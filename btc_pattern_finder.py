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
    
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.feature_extractor = None  # Will be initialized on demand
        
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
                    
                    match_results.append({
                        "distance": float(match_dist),
                        "start_time": datetime.fromtimestamp(match_start / 1000).isoformat(),
                        "end_time": datetime.fromtimestamp(match_end / 1000).isoformat(),
                        "timestamp": int(match_start),
                        "pattern_data": match_data[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records"),
                        "label": f"{symbol} from {match_datetime.strftime('%Y-%m-%d %H:%M')}"
                    })
                
        # Sort by distance (similarity)
        sorted_results = sorted(match_results, key=lambda x: x["distance"])
        
        # Add debugging info to help understand filtering
        # Also apply a limit to the results based on the original max_matches request
        final_results = sorted_results[:max_matches] if len(sorted_results) > max_matches else sorted_results
        
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
            "debug_info": {
                "requested_matches": max_matches,
                "stumpy_matches_found": len(matches) if matches else 0,
                "unique_matches": len(sorted_results),  # Matches after time-proximity filtering
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
                    
                    match_results.append({
                        "distance": float(match_dist),
                        "start_time": datetime.fromtimestamp(match_start / 1000).isoformat(),
                        "end_time": datetime.fromtimestamp(match_end / 1000).isoformat(),
                        "timestamp": int(match_start),
                        "pattern_data": match_data[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records"),
                        "label": f"{symbol} from {match_datetime.strftime('%Y-%m-%d %H:%M')}"
                    })
        
        # Sort by distance
        sorted_results = sorted(match_results, key=lambda x: x["distance"])
        
        # Apply max_matches limit
        final_results = sorted_results[:max_matches] if len(sorted_results) > max_matches else sorted_results
        
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
            "debug_info": {
                "requested_matches": max_matches,
                "feature_matches_found": len(matches) if matches else 0,
                "unique_matches": len(sorted_results),
                "final_matches": len(final_results)
            }
        }