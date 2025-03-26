"""
Feature extraction module for price pattern analysis.
Extracts statistical and technical features from price data for
dimensionality reduction and pattern family identification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PriceFeatureExtractor:
    """Extracts features from OHLC price data for pattern analysis."""
    
    def __init__(self, window_size: int = 30, n_components: int = 2):
        """
        Initialize the feature extractor.
        
        Args:
            window_size: Size of the sliding window for feature extraction
            n_components: Number of dimensions to reduce to
        """
        self.window_size = window_size
        self.method = 'pca'  # Only PCA is supported
        self.n_components = n_components
        self.scaler = StandardScaler()
        
    def extract_features(self, ohlc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from OHLC price data.
        
        Args:
            ohlc_df: DataFrame with 'open', 'high', 'low', 'close' columns
            
        Returns:
            DataFrame of extracted features
        """
        # Create a copy to avoid modifying the original
        price_data = ohlc_df[['open', 'high', 'low', 'close']].copy()
        
        # Create a normalized version of the price data for shape-focused features
        norm_price_data = price_data.copy()
        
        # Scale by the range of the entire pattern to normalize for shape matching
        price_range = norm_price_data['high'].max() - norm_price_data['low'].min()
        if price_range > 0:  # Avoid division by zero
            baseline = norm_price_data['low'].min()
            norm_price_data['open'] = (norm_price_data['open'] - baseline) / price_range
            norm_price_data['high'] = (norm_price_data['high'] - baseline) / price_range
            norm_price_data['low'] = (norm_price_data['low'] - baseline) / price_range
            norm_price_data['close'] = (norm_price_data['close'] - baseline) / price_range
            
        # Calculate returns and price relationships
        price_data['returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
        price_data['high_low_range'] = price_data['high'] - price_data['low']
        price_data['body_size'] = abs(price_data['close'] - price_data['open'])
        price_data['upper_shadow'] = price_data.apply(
            lambda x: x['high'] - max(x['open'], x['close']), axis=1
        )
        price_data['lower_shadow'] = price_data.apply(
            lambda x: min(x['open'], x['close']) - x['low'], axis=1
        )
        
        # Add body position features (where is the body within the range)
        price_data['body_position'] = (price_data['close'] + price_data['open']) / 2 - price_data['low']
        price_data['body_position_rel'] = np.where(
            price_data['high_low_range'] > 0,
            price_data['body_position'] / price_data['high_low_range'],
            0.5  # Default to middle if there's no range
        )
        
        # Add body direction (bullish/bearish)
        price_data['body_direction'] = np.sign(price_data['close'] - price_data['open'])
        
        # Drop NaN values from returns calculation
        price_data = price_data.dropna()
        
        # Calculate statistical features
        features = {}
        
        try:
            # Price action features - with added error checking
            features['trend'] = (price_data['close'].iloc[-1] / price_data['close'].iloc[0]) - 1 if len(price_data) > 0 else 0
            features['volatility'] = price_data['returns'].std() * np.sqrt(len(price_data)) if len(price_data) > 0 else 0
            
            # Calculate max drawdown
            if len(price_data) > 0:
                rolling_max = price_data['close'].cummax()
                drawdown = (price_data['close'] / rolling_max - 1)
                features['max_drawdown'] = drawdown.min()
            else:
                features['max_drawdown'] = 0
            
            # Momentum indicators - with safe shift operations
            if len(price_data) > 1:
                price_data['momentum_1'] = price_data['close'] / price_data['close'].shift(1) - 1
            else:
                price_data['momentum_1'] = pd.Series([0] * len(price_data))
                
            if len(price_data) > 3:
                price_data['momentum_3'] = price_data['close'] / price_data['close'].shift(3) - 1
            else:
                price_data['momentum_3'] = pd.Series([0] * len(price_data))
                
            if len(price_data) > 5:
                price_data['momentum_5'] = price_data['close'] / price_data['close'].shift(5) - 1
            else:
                price_data['momentum_5'] = pd.Series([0] * len(price_data))
            
            # Price velocity and acceleration
            price_data['velocity'] = price_data['close'].diff().fillna(0)
            price_data['acceleration'] = price_data['velocity'].diff().fillna(0)
            
            # Candlestick features - with division by zero protection
            hlr_mean = price_data['high_low_range'].mean()
            if hlr_mean > 0:
                features['avg_body_ratio'] = price_data['body_size'].mean() / hlr_mean
                features['upper_shadow_ratio'] = price_data['upper_shadow'].mean() / hlr_mean
                features['lower_shadow_ratio'] = price_data['lower_shadow'].mean() / hlr_mean
            else:
                features['avg_body_ratio'] = 0
                features['upper_shadow_ratio'] = 0
                features['lower_shadow_ratio'] = 0
                
            # Consistency features - how consistent are the candle visual properties
            if len(price_data) > 1:
                # Body size consistency
                if price_data['body_size'].mean() > 0:
                    features['body_size_consistency'] = price_data['body_size'].std() / price_data['body_size'].mean()
                else:
                    features['body_size_consistency'] = 0
                
                # Upper/lower shadow consistency
                if price_data['upper_shadow'].mean() > 0:
                    features['upper_shadow_consistency'] = price_data['upper_shadow'].std() / price_data['upper_shadow'].mean()
                else:
                    features['upper_shadow_consistency'] = 0
                    
                if price_data['lower_shadow'].mean() > 0:
                    features['lower_shadow_consistency'] = price_data['lower_shadow'].std() / price_data['lower_shadow'].mean()
                else:
                    features['lower_shadow_consistency'] = 0
                
                # Body direction consistency - measures if candles are consistently bullish/bearish
                features['direction_consistency'] = abs(price_data['body_direction'].sum()) / len(price_data)
                
                # Body position consistency - where bodies appear within the candle ranges
                features['body_position_consistency'] = price_data['body_position_rel'].std()
            else:
                features['body_size_consistency'] = 0
                features['upper_shadow_consistency'] = 0
                features['lower_shadow_consistency'] = 0
                features['direction_consistency'] = 0
                features['body_position_consistency'] = 0
            
            # Statistical features for returns - safely handle empty dataframes
            for col in ['returns', 'high_low_range', 'body_size', 'momentum_1', 'velocity', 'acceleration']:
                features[f'{col}_mean'] = price_data[col].mean() if not price_data[col].empty else 0
                features[f'{col}_std'] = price_data[col].std() if not price_data[col].empty else 0
                # For performance, only use mean and std - skip skewness and kurtosis
                # These are the most important features and less likely to cause numerical errors
            
            # Pattern complexity - count direction changes (safely)
            for col in ['close', 'velocity']:
                try:
                    sign_changes = ((np.sign(price_data[col].diff().fillna(0)) != 
                                   np.sign(price_data[col].diff().shift(1).fillna(0))) & 
                                   (np.sign(price_data[col].diff().fillna(0)) != 0)).sum()
                    features[f'{col}_sign_changes'] = sign_changes / len(price_data) if len(price_data) > 0 else 0
                except:
                    features[f'{col}_sign_changes'] = 0
            
            # Autocorrelation features (safely)
            for lag in [1, 2, 3, 5]:
                try:
                    if len(price_data) > lag + 1:
                        features[f'autocorr_{lag}'] = price_data['returns'].autocorr(lag)
                    else:
                        features[f'autocorr_{lag}'] = 0
                except:
                    features[f'autocorr_{lag}'] = 0
                    
            # Capture relationships between OHLC components using normalized data
            if len(norm_price_data) > 1:
                # Correlations between price components 
                features['open_close_correlation'] = norm_price_data['open'].corr(norm_price_data['close'])
                features['high_low_correlation'] = norm_price_data['high'].corr(norm_price_data['low'])
                
                # Pattern shape features with polynomial fit on normalized data
                x = np.arange(len(norm_price_data))
                try:
                    # Quadratic fit to capture overall curve shape
                    close_fit = np.polyfit(x, norm_price_data['close'], 2)
                    features['close_shape_a'] = close_fit[0]  # Quadratic coefficient (curvature)
                    features['close_shape_b'] = close_fit[1]  # Linear coefficient (slope)
                    features['close_shape_c'] = close_fit[2]  # Constant term
                    
                    # Also fit high and low to capture range dynamics
                    high_fit = np.polyfit(x, norm_price_data['high'], 2)
                    features['high_shape_a'] = high_fit[0]
                    
                    low_fit = np.polyfit(x, norm_price_data['low'], 2)
                    features['low_shape_a'] = low_fit[0]
                    
                    # Range expansion/contraction
                    range_data = norm_price_data['high'] - norm_price_data['low']
                    range_fit = np.polyfit(x, range_data, 1)
                    features['range_trend'] = range_fit[0]  # Positive means expanding range
                    
                except:
                    features['close_shape_a'] = 0
                    features['close_shape_b'] = 0
                    features['close_shape_c'] = 0
                    features['high_shape_a'] = 0
                    features['low_shape_a'] = 0
                    features['range_trend'] = 0
            else:
                features['open_close_correlation'] = 0
                features['high_low_correlation'] = 0
                features['close_shape_a'] = 0
                features['close_shape_b'] = 0
                features['close_shape_c'] = 0
                features['high_shape_a'] = 0
                features['low_shape_a'] = 0
                features['range_trend'] = 0
        
        except Exception as e:
            print(f"Error calculating features: {str(e)}")
            # Provide basic fallback features if calculation fails
            features = {
                'trend': 0, 
                'volatility': 0,
                'basic_mean': np.mean(price_data['close']) if len(price_data) > 0 else 0,
                'basic_std': np.std(price_data['close']) if len(price_data) > 0 else 0
            }
        
        # Return as a DataFrame with one row
        return pd.DataFrame([features])
    
    def extract_features_from_windows(self, 
                                      ohlc_df: pd.DataFrame, 
                                      window_size: Optional[int] = None, 
                                      stride: int = 1) -> List[pd.DataFrame]:
        """
        Extract features from sliding windows in price data.
        
        Args:
            ohlc_df: DataFrame with 'open', 'high', 'low', 'close' columns
            window_size: Size of sliding window (default: self.window_size)
            stride: Steps between windows
            
        Returns:
            List of feature DataFrames for each window
        """
        if window_size is None:
            window_size = self.window_size
            
        windows = []
        for i in range(0, len(ohlc_df) - window_size + 1, stride):
            window = ohlc_df.iloc[i:i+window_size]
            windows.append(window)
            
        all_features = []
        for window in windows:
            window_features = self.extract_features(window)
            all_features.append(window_features)
            
        return all_features
    
    def reduce_dimensions(self, features_df: pd.DataFrame) -> Dict:
        """
        Apply dimensionality reduction to extracted features.
        
        Args:
            features_df: DataFrame of features to reduce
            
        Returns:
            Dictionary with reduced data and metadata
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Apply PCA dimensionality reduction method
        print(f"Using PCA dimensionality reduction")
        
        # PCA reduction
        reducer = PCA(n_components=self.n_components)
        reduced_data = reducer.fit_transform(scaled_features)
        explained_variance = reducer.explained_variance_ratio_.sum()
        
        # Calculate feature importance
        components = reducer.components_
        feature_importance = {
            features_df.columns[i]: np.abs(components[:, i]).sum() 
            for i in range(features_df.shape[1])
        }
        
        # Create result DataFrame
        result_df = pd.DataFrame(
            reduced_data, 
            columns=[f'component_{i+1}' for i in range(self.n_components)]
        )
        
        return {
            'reduced_data': result_df,
            'explained_variance': explained_variance,
            'feature_importance': feature_importance
        }
        
    def find_similar_patterns(self, 
                             source_ohlc: pd.DataFrame, 
                             historical_ohlc: pd.DataFrame, 
                             window_size: Optional[int] = None,
                             stride: int = 1,
                             top_n: int = 10) -> Tuple[List[Tuple[float, int]], Dict]:
        """
        Find patterns similar to source pattern using PCA feature extraction.
        
        Args:
            source_ohlc: Source pattern OHLC data
            historical_ohlc: Historical OHLC data to search
            window_size: Size of the pattern window
            stride: Steps between windows
            top_n: Number of most similar patterns to return
            
        Returns:
            Tuple of (distances, visualization_data)
            - distances: List of (distance, index) tuples sorted by similarity
            - visualization_data: Dict with data for visualizing feature space
        """
        print(f"Starting feature-based pattern search...")
        print(f"Source pattern size: {len(source_ohlc)} candles")
        print(f"Historical data size: {len(historical_ohlc)} candles")
        if window_size is None:
            window_size = self.window_size
            
        # Extract features from source pattern
        source_features = self.extract_features(source_ohlc)
        
        # Extract features from all historical windows
        windows = []
        window_indices = []
        
        for i in range(0, len(historical_ohlc) - window_size + 1, stride):
            window = historical_ohlc.iloc[i:i+window_size]
            windows.append(window)
            window_indices.append(i)
            
        # Extract features from all windows
        print(f"Extracting features from {len(windows)} windows...")
        all_features = []
        for i, window in enumerate(windows):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i}/{len(windows)} windows...")
            window_features = self.extract_features(window)
            all_features.append(window_features)
            
        if not all_features:
            print("No features extracted, returning empty results")
            return [], {}
            
        print(f"Combining features from {len(all_features)} windows...")
        # Combine all features including source
        combined_features = pd.concat(all_features, ignore_index=True)
        all_combined = pd.concat([source_features, combined_features], ignore_index=True)
        
        print(f"Applying PCA dimensionality reduction to {all_combined.shape[0]} samples with {all_combined.shape[1]} features...")
        # Apply dimensionality reduction
        reduction_result = self.reduce_dimensions(all_combined)
        
        # Get reduced coordinates
        reduced_data = reduction_result['reduced_data']
        source_reduced = reduced_data.iloc[0].values
        historical_reduced = reduced_data.iloc[1:].values
        
        print(f"Calculating distances between source pattern and {len(historical_reduced)} windows...")
        # Calculate Euclidean distances in reduced space - using vectorized operations for speed
        try:
            # Fast vectorized calculation
            diffs = historical_reduced - source_reduced
            distances = [(np.sqrt((diff**2).sum()), idx) for diff, idx in zip(diffs, window_indices)]
            print(f"Distance calculation complete: found {len(distances)} matches")
        except Exception as e:
            print(f"Error during vectorized distance calculation: {str(e)}")
            # Fall back to loop implementation
            print("Falling back to loop implementation...")
            distances = []
            for i, point in enumerate(historical_reduced):
                if i % 1000 == 0 and i > 0:
                    print(f"Processed {i}/{len(historical_reduced)} points...")
                dist = np.sqrt(((source_reduced - point) ** 2).sum())
                window_idx = window_indices[i]
                distances.append((dist, window_idx))
            
        print(f"Sorting {len(distances)} distances...")
        # Sort by distance (ascending)
        sorted_distances = sorted(distances, key=lambda x: x[0])
        
        # Return top N matches and visualization data
        vis_data = {
            'reduced_data': reduction_result['reduced_data'],
            'explained_variance': reduction_result['explained_variance'],
            'feature_importance': reduction_result['feature_importance'],
            'window_indices': [0] + window_indices,  # Add source index
            'source_index': 0
        }
        
        return sorted_distances[:top_n], vis_data