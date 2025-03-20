import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple

def plot_pattern_match(
    source_pattern: pd.DataFrame,
    match_pattern: pd.DataFrame,
    title: str = "BTC Pattern Match Comparison",
    dark_mode: bool = False
) -> plt.Figure:
    """
    Create a visualization comparing source pattern with a matched pattern
    
    Args:
        source_pattern: DataFrame with source pattern OHLCV data
        match_pattern: DataFrame with matched pattern OHLCV data
        title: Plot title
        dark_mode: Whether to use dark theme for the plot
        
    Returns:
        Matplotlib Figure object
    """
    # Set dark theme if requested
    if dark_mode:
        plt.style.use('dark_background')
        bg_color = '#121212'
        grid_color = '#333333'
        text_color = '#e0e0e0'
        source_color = '#2d7bf4'
        match_color = '#26a69a'
        future_color = '#ef5350'
        line_color = '#ffeb3b'
    else:
        plt.style.use('default')
        bg_color = 'white'
        grid_color = '#dddddd'
        text_color = 'black'
        source_color = 'blue'
        match_color = 'green'
        future_color = 'purple'
        line_color = 'red'
    
    # Create a figure with 2 rows and 1 column
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    if dark_mode:
        fig.patch.set_facecolor(bg_color)
        for ax in axes:
            ax.set_facecolor(bg_color)
    
    # Normalize both patterns for better comparison
    source_close = source_pattern["close"].values
    match_close = match_pattern["close"].values
    
    # Z-normalize (same normalization used by stumpy internally)
    def z_normalize(data):
        return (data - np.mean(data)) / np.std(data)
    
    source_norm = z_normalize(source_close)
    match_norm = z_normalize(match_close)
    
    # Determine pattern length (for vertical line positioning)
    pattern_length = len(source_close)
    
    # Convert timestamps to datetime for better x-axis
    source_times = [datetime.fromtimestamp(ts/1000) for ts in source_pattern["timestamp"]]
    match_times = [datetime.fromtimestamp(ts/1000) for ts in match_pattern["timestamp"]]
    
    # Plot source pattern
    axes[0].plot(source_times, source_norm, label="Source Pattern", color=source_color, linewidth=2)
    axes[0].set_title(f"Source Pattern: Current BTC", fontsize=14, color=text_color)
    axes[0].grid(True, alpha=0.3, color=grid_color)
    axes[0].legend(facecolor=bg_color if dark_mode else None, 
                   labelcolor=text_color if dark_mode else None)
    
    # Plot match pattern with separation line
    axes[1].plot(match_times, match_norm, label="Matched Pattern", color=match_color, linewidth=2)
    
    # Add vertical line to separate pattern from future data
    if len(match_times) > pattern_length:
        prediction_start = match_times[pattern_length-1]
        axes[1].axvline(x=prediction_start, color=line_color, linestyle="--", 
                        label="Pattern End/Prediction Start")
        
        # Plot the future part with different color
        future_times = match_times[pattern_length-1:]
        future_values = match_norm[pattern_length-1:]
        axes[1].plot(future_times, future_values, color=future_color, linewidth=2, 
                    label="What Happened After")
    
    match_date = match_times[0].strftime("%Y-%m-%d")
    axes[1].set_title(f"Matched Pattern: BTC from {match_date}", fontsize=14, color=text_color)
    axes[1].grid(True, alpha=0.3, color=grid_color)
    axes[1].legend(facecolor=bg_color if dark_mode else None, 
                   labelcolor=text_color if dark_mode else None)
    
    # Set axis labels color
    for ax in axes:
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
    
    # Overall title
    plt.suptitle(title, fontsize=16, color=text_color)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig

def plot_multiple_matches(
    source_pattern: pd.DataFrame,
    matches: List[Dict],
    max_plots: int = 3,
    title: str = "BTC Historical Pattern Matches",
    dark_mode: bool = False
) -> plt.Figure:
    """
    Create a visualization comparing source pattern with multiple matches
    
    Args:
        source_pattern: DataFrame with source pattern OHLCV data
        matches: List of dictionaries with match data
        max_plots: Maximum number of matches to plot
        title: Plot title
        dark_mode: Whether to use dark theme for the plot
        
    Returns:
        Matplotlib Figure object
    """
    # Set dark theme if requested
    if dark_mode:
        plt.style.use('dark_background')
        bg_color = '#121212'
        grid_color = '#333333'
        text_color = '#e0e0e0'
        source_color = '#2d7bf4'
        match_color = '#26a69a'
        future_color = '#ef5350'
        line_color = '#ffeb3b'
    else:
        plt.style.use('default')
        bg_color = 'white'
        grid_color = '#dddddd'
        text_color = 'black'
        source_color = 'blue'
        match_color = 'green'
        future_color = 'purple'
        line_color = 'red'
    
    num_matches = min(len(matches), max_plots)
    
    # Create a figure with multiple rows
    fig, axes = plt.subplots(num_matches + 1, 1, figsize=(12, 4 * (num_matches + 1)), sharex=False)
    if dark_mode:
        fig.patch.set_facecolor(bg_color)
        if num_matches == 0:  # Handle single axis case
            axes.set_facecolor(bg_color)
        else:
            for ax in axes:
                ax.set_facecolor(bg_color)
    
    # Convert to array if it's a single axis
    axes = np.array([axes]) if num_matches == 0 else axes
    
    # Normalize source pattern
    source_close = source_pattern["close"].values
    
    def z_normalize(data):
        return (data - np.mean(data)) / np.std(data)
    
    source_norm = z_normalize(source_close)
    
    # Convert timestamps to datetime for better x-axis
    source_times = [datetime.fromtimestamp(ts/1000) for ts in source_pattern["timestamp"]]
    
    # Plot source pattern
    axes[0].plot(source_times, source_norm, label="Source Pattern", color=source_color, linewidth=2)
    axes[0].set_title(f"Source Pattern: Current BTC", fontsize=14, color=text_color)
    axes[0].grid(True, alpha=0.3, color=grid_color)
    axes[0].legend(facecolor=bg_color if dark_mode else None, 
                   labelcolor=text_color if dark_mode else None)
    
    # Plot each match
    for i in range(num_matches):
        match = matches[i]
        match_data = pd.DataFrame(match["pattern_data"])
        
        match_close = match_data["close"].values
        match_norm = z_normalize(match_close)
        
        # Convert timestamps to datetime
        match_times = [datetime.fromtimestamp(ts/1000) for ts in match_data["timestamp"]]
        
        # Pattern length for vertical line
        pattern_length = len(source_close)
        
        # Plot match pattern
        axes[i+1].plot(match_times[:pattern_length], match_norm[:pattern_length], 
                      label=f"Match #{i+1} (Distance: {match['distance']:.4f})", 
                      color=match_color, linewidth=2)
        
        # Add vertical line to separate pattern from future data
        if len(match_times) > pattern_length:
            prediction_start = match_times[pattern_length-1]
            axes[i+1].axvline(x=prediction_start, color=line_color, linestyle="--", 
                            label="Pattern End/Prediction Start")
            
            # Plot the future part with different color
            future_times = match_times[pattern_length-1:]
            future_values = match_norm[pattern_length-1:]
            axes[i+1].plot(future_times, future_values, color=future_color, linewidth=2, 
                          label="What Happened After")
        
        match_date = match_times[0].strftime("%Y-%m-%d")
        axes[i+1].set_title(f"Match #{i+1}: BTC from {match_date} (Distance: {match['distance']:.4f})", 
                           fontsize=14, color=text_color)
        axes[i+1].grid(True, alpha=0.3, color=grid_color)
        axes[i+1].legend(facecolor=bg_color if dark_mode else None, 
                         labelcolor=text_color if dark_mode else None)
    
    # Set axis labels color
    for ax in axes:
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
    
    # Overall title
    plt.suptitle(title, fontsize=16, color=text_color)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


def create_feature_space_visualization(
    vis_data: Dict,
    feature_method: str,
    match_indices: List[int],
    dark_mode: bool = True
) -> go.Figure:
    """
    Create a visualization of the feature space after dimensionality reduction.
    
    Args:
        vis_data: Dictionary with visualization data
        feature_method: Method used (always 'pca')
        match_indices: Indices of matched patterns in the dataset
        dark_mode: Use dark theme if True
        
    Returns:
        Plotly figure object
    """
    # Set colors based on theme
    if dark_mode:
        bg_color = "#1e1e1e"  # Match the app background color
        grid_color = "#333333"
        text_color = "#cccccc"  # Match the app text color
        source_color = "#2d7bf4"  # Blue
        match_color = "#26a69a"   # Teal
        other_color = "#666666"   # Gray
    else:
        bg_color = "white"
        grid_color = "#dddddd"
        text_color = "black"
        source_color = "blue"
        match_color = "green"
        other_color = "#999999"   # Gray
    
    # Extract data from vis_data
    reduced_data = vis_data['reduced_data']
    source_index = vis_data['source_index']
    
    # Create point types for coloring
    point_types = ['Other'] * len(reduced_data)
    point_types[source_index] = 'Source Pattern'
    
    for idx in match_indices:
        if idx < len(point_types):
            point_types[idx] = 'Match'
    
    # Create dataframe for plotting
    plot_df = reduced_data.copy()
    plot_df['point_type'] = point_types
    
    # Create figure based on number of components
    n_components = reduced_data.shape[1]
    
    if n_components >= 3:
        # 3D scatter plot for 3+ components
        fig = px.scatter_3d(
            plot_df,
            x='component_1',
            y='component_2',
            z='component_3',
            color='point_type',
            color_discrete_map={
                'Source Pattern': source_color,
                'Match': match_color,
                'Other': other_color
            },
            opacity=0.7
        )
    else:
        # 2D scatter plot for 2 components
        fig = px.scatter(
            plot_df,
            x='component_1',
            y='component_2',
            color='point_type',
            color_discrete_map={
                'Source Pattern': source_color,
                'Match': match_color,
                'Other': other_color
            },
            opacity=0.7
        )
    
    # Configure layout
    title = f"{feature_method.upper()} Feature Space"
    if 'explained_variance' in vis_data and vis_data['explained_variance'] is not None:
        title += f" (Explained Variance: {vis_data['explained_variance']:.2%})"
    
    fig.update_layout(
        title=title,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color="#cccccc"),  # Match the app text color
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,  # Remove the legend title
            font=dict(color="#cccccc"),  # Ensure legend text matches app color
            itemsizing="constant"
        ),
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    # Update legend text color specifically
    fig.for_each_trace(
        lambda trace: trace.update(
            showlegend=True,
            legendgroup=trace.name,
            name=trace.name
        )
    )
    
    return fig


def create_feature_importance_chart(
    feature_importance: Dict[str, float],
    top_n: int = 15,
    dark_mode: bool = True
) -> go.Figure:
    """
    Create a bar chart showing the most important features.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        top_n: Number of top features to show
        dark_mode: Use dark theme if True
        
    Returns:
        Plotly figure object
    """
    if not feature_importance:
        return None
        
    # Set colors based on theme
    if dark_mode:
        bg_color = "#1e1e1e"  # Match the app background color
        grid_color = "#333333" 
        text_color = "#cccccc"  # Match the app text color
        bar_color = "#2d7bf4"
    else:
        bg_color = "white"
        grid_color = "#dddddd"
        text_color = "black"
        bar_color = "blue"
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    # Create dataframe for plotting
    feature_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    
    # Create horizontal bar chart
    fig = px.bar(
        feature_df,
        y='Feature',
        x='Importance',
        orientation='h',
        color_discrete_sequence=[bar_color]
    )
    
    # Configure layout
    fig.update_layout(
        title="Top Features by Importance",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis_title="Importance Score",
        yaxis_title="",
        height=min(100 + 20 * len(top_features), 600),  # Dynamic height based on feature count
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig