import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

from btc_pattern_finder import BTCPatternFinder
from data_provider import BinanceDataProvider, CompositeBTCDataProvider

# Minimum year for searching
MIN_YEAR = 2017

# Set page config
st.set_page_config(
    page_title="BTC Pattern Finder",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the custom ProtoMono-Light font from file
import base64
import os

# Get the font file path
font_path = os.path.join(os.path.dirname(__file__), 'ProtoMono-Light.otf')

# Load and encode the font
with open(font_path, 'rb') as font_file:
    font_base64 = base64.b64encode(font_file.read()).decode()

# Load custom font and apply clean, modern CSS with chart spacing adjustments
st.markdown(f"""
<style>
    /* Custom Font Loading */
    @font-face {{
        font-family: 'ProtoMono-Light';
        src: url('data:font/otf;base64,{font_base64}') format('opentype');
        font-weight: normal;
        font-style: normal;
    }}
    
    /* Global font settings - use ProtoMono-Light everywhere */
    * {{
        font-family: 'ProtoMono-Light', monospace !important;
    }}
    
    /* Consistent font size and pale grey text color for better readability */
    body, p, div, span, label, select, input, .element-container {{
        font-size: 1rem !important;
        color: #cccccc !important;
    }}
    
    /* Header styles */
    h1 {{font-size: 1rem !important; color: #dddddd !important;}}
    h2 {{font-size: 1rem !important; color: #dddddd !important;}}
    h3 {{font-size: 1rem !important; color: #dddddd !important;}}
    
    /* Button styling - lighter buttons than background */
    .stButton button {{
        width: 100%;
        color: #333333 !important;
        background-color: #2c2c2c !important;
        border-color: #444444 !important;
    }}
    
    /* Button hover state */
    .stButton button:hover {{
        background-color: #3a3a3a !important;
        border-color: #555555 !important;
    }}
    
    /* Container adjustments with minimal margins and padding */
    .main .block-container {{
        padding: 0.2rem !important; /* Extremely reduced padding */
        padding-top: 0 !important; /* Remove top padding completely */
        padding-bottom: 20px !important; /* Reduced bottom padding */
        max-width: 100% !important;
        margin: 0 auto !important;
    }}
    
    /* Force main container to start at the very top */
    .main {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}
    
    /* Eliminate padding from app container */
    .appview-container {{
        padding-top: 0 !important;
    }}
    
    #MainMenu {{display: none;}} footer {{display: none;}} header {{display: none;}}
    .stDateInput {{width: 100%;}}
    /* Aggressively reduce spacing for headers */
    h1 {{margin-top: 0.2rem !important; margin-bottom: 0.1rem !important; font-size: 1rem !important;}}
    h2, h3 {{margin-top: 0.3rem !important; margin-bottom: 0.1rem !important;}}
    
    /* Aggressively reduce spacing for Streamlit elements */
    .stSelectbox, .stMultiselect {{margin-bottom: 0.3rem !important; margin-top: 0 !important;}}
    div[data-testid="stVerticalBlock"] > div {{padding-top: 0 !important; padding-bottom: 0 !important; margin-top: 0 !important; margin-bottom: 0 !important;}}
    .element-container {{margin-top: 0 !important; padding-top: 0 !important; margin-bottom: 0.2rem !important;}}
    
    /* Target specific spacing between controls and title */
    div[data-testid="stHorizontalBlock"] {{
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }}
    .st-emotion-cache-1wmy9hl {{overflow-x: hidden !important;}}
    
    /* Custom tooltip styling */
    div[data-testid="stTooltipIcon"] span {{
        background-color: #262730;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.2rem;
    }}
    
    /* Target chart containers to reduce vertical spacing */
    .stPlotlyChart {{
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }}
    
    /* Target the actual Plotly graphs */
    .js-plotly-plot, .plot-container {{
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }}
    
    /* Drastically reduce whitespace */
    .element-container {{
        margin-bottom: 5px !important; /* Reduced from 10px */
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }}
    
    /* Eliminate colored debug backgrounds and match chart background */
    body, .stPlotlyChart, .js-plotly-plot, .plot-container, .element-container, .block-container, .main {{
        background-color: #1e1e1e !important; /* Match the chart background color */
    }}
    
    /* Add spacing around charts with reduced side margins */
    .stPlotlyChart {{
        padding: 5px !important;
        padding-bottom: 20px !important; /* Extra padding at bottom for x-axis labels */
        padding-right: 8px !important; /* Reduced right padding by 50% */
        margin-bottom: 15px !important;
        margin-top: 10px !important;
        margin-right: 5px !important; /* Reduced right margin by 50% */
        border: none; /* Remove border entirely */
        overflow: visible !important; /* Allow content to overflow for axis labels */
        box-sizing: content-box !important; /* Ensure proper size calculation */
        max-width: 100% !important; /* Reduced adjustment by 50% */
    }}
    
    /* Make axis labels and ticks consistent */
    .xtick text, .ytick text {{
        color: #999999 !important;
    }}
    
    /* Ensure grid lines are visible */
    .gridlayer path {{
        stroke: #333333 !important;
    }}
    
    /* Chart spacing and visibility */
    .stPlotlyChart {{
        border: none; /* Remove border entirely */
        padding: 8px;
        padding-bottom: 20px;
        margin-bottom: 15px;
        overflow: visible !important;
    }}
    
    /* Make sure plotly objects appear */
    .plot-container, .plotly, .js-plotly-plot, .svg-container {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    
    /* Ensure canvas is visible */
    canvas, .main-svg {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    
    /* Remove unwanted subplot titles */
    .gtitle, .g-gtitle {{
        display: none !important;
    }}
    
    /* Target text elements between charts */
    p, div.markdown-text-container {{
        margin-bottom: 8px !important;
        margin-top: 8px !important;
        padding-left: 5px !important;
    }}
    
    /* Add space at the bottom of the page */
    .main .block-container {{
        padding-bottom: 40px !important;
    }}
    
    /* Provide a bit more spacing on the right for the scrollbar */
    body {{
        padding-right: 15px !important;
    }}
    
    /* Slider styling - leaving default values */
    
    /* Highlight select options on hover with light blue */
    div[data-baseweb="select-option"]:hover {{
        background-color: rgba(66, 135, 245, 0.2) !important;
    }}
    
    /* Style select boxes to match button styling */
    .stSelectbox [data-baseweb="select"] div, 
    .stMultiselect [data-baseweb="select"] div {{
        background-color: #2c2c2c !important;
        border-color: #444444 !important;
        color: #cccccc !important;
    }}
    
    .stSelectbox [data-baseweb="select"] div:hover,
    .stMultiselect [data-baseweb="select"] div:hover {{
        background-color: #3a3a3a !important;
        border-color: #555555 !important;
    }}
    
    /* Style dropdown menus to match buttons - more aggressive styling */
    div[data-baseweb="popover"] div[data-baseweb="menu"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] {{
        background-color: #222222 !important;
        border-color: #444444 !important;
        color: #cccccc !important;
    }}
    
    div[data-baseweb="select-option"],
    div[data-baseweb="menu"] div,
    div[data-baseweb="menu"] li,
    div[role="option"],
    li[role="option"] {{
        background-color: #222222 !important;
        color: #cccccc !important;
    }}
    
    div[data-baseweb="select-option"]:hover,
    div[role="option"]:hover,
    li[role="option"]:hover {{
        background-color: #3a3a3a !important;
    }}
    
    /* Force select dropdown color */
    div[data-baseweb="select"] div, 
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg,
    div[data-baseweb="select"] * {{
        color: #cccccc !important;
        fill: #cccccc !important;
    }}
    
    /* Make dropdown menu items selected state more visible */
    div[aria-selected="true"],
    li[aria-selected="true"] {{
        background-color: rgba(66, 135, 245, 0.3) !important;
    }}
    
    /* Make date selector buttons more compact but preserve font size */
    div[data-testid="stSelectbox"] {{
        max-width: 85%;
        width: 85%;
    }}
    
    /* Specific styling for date selector boxes in search period */
    #start_year, #start_month, #start_day, #end_year, #end_month, #end_day {{
        max-width: 75%;  /* Reduce width by 15% */
        width: 75%;
    }}
    
    /* Ensure font size remains consistent with the rest of the site */
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] div,
    div[data-baseweb="select"] span, 
    div[data-baseweb="select"] div {{
        font-size: 1rem !important;
    }}
    
    /* Force the selection boxes to be smaller */
    div[role="listbox"] {{
        max-width: 90% !important;
    }}
    
    /* Reduce width of date selectors container */
    div.stSelectbox {{
        max-width: 90% !important;
    }}
</style>
""", unsafe_allow_html=True)

# Remove the scroll container and JavaScript - this didn't work
# We're focusing on reducing vertical space instead

# Initialize session state variables
if 'btc_data' not in st.session_state:
    st.session_state.btc_data = None
    
if 'selected_range' not in st.session_state:
    st.session_state.selected_range = {'start_idx': None, 'end_idx': None}
    
if 'interval' not in st.session_state:
    st.session_state.interval = '30m'  # Default to 30 minutes timeframe
    
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
    
if 'candle_style' not in st.session_state:
    st.session_state.candle_style = {
        'increasing_color': '#999999',         # Medium grey for up candles (matches axis text)
        'decreasing_color': 'rgba(0,0,0,0)',   # Transparent fill for down candles
        'increasing_line_color': '#999999',    # Medium grey outline for up candles
        'decreasing_line_color': '#999999',    # Medium grey outline for down candles
        'background_color': '#1e1e1e',
        'grid_color': '#333333',
        'selected_color': 'rgba(65, 105, 225, 0.3)'
    }
    
if 'search_range' not in st.session_state:
    st.session_state.search_range = {
        'start_date': datetime(MIN_YEAR, 1, 1),
        'end_date': datetime.now()
    }
    
# Initialize coin selection if not already set
if 'selected_coin' not in st.session_state:
    st.session_state.selected_coin = 'BTC'  # Default to Bitcoin
    
# Initialize feature extraction related session state variables
if 'analysis_method' not in st.session_state:
    st.session_state.analysis_method = 'Matrix Profile'  # Default to Matrix Profile approach
    
if 'feature_method' not in st.session_state:
    st.session_state.feature_method = 'PCA'  # Default dimensionality reduction method
    
if 'n_components' not in st.session_state:
    st.session_state.n_components = 2  # Default number of components

# Initialize data provider - don't cache to avoid timeframe mixups
def get_data_provider():
    return CompositeBTCDataProvider([BinanceDataProvider()])

# Don't cache pattern finder to avoid mixing up different timeframes
def get_pattern_finder():
    provider = get_data_provider()
    return BTCPatternFinder(provider)

# Data loading function
async def load_market_data(symbol="BTC", interval='30m', days=30):
    provider = get_data_provider()
    current_time = datetime.now()
    start_time = (current_time - timedelta(days=days)).isoformat()
    
    df = await provider.get_historical_ohlcv(
        symbol=symbol, 
        interval=interval,
        start_time=start_time,
        end_time=current_time.isoformat()
    )
    
    # Convert timestamp to datetime for better plotting
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['idx'] = range(len(df))
    
    return df

# Create a Plotly candlestick chart with selection capabilities
def create_candlestick_chart(df, selected_range=None, style=None):
    if df is None or df.empty:
        return None
    
    if style is None:
        style = st.session_state.candle_style
    
    # Create figure - removed volume chart
    fig = make_subplots(
        rows=1, 
        cols=1
    )
    
    # Extract selected range indices
    start_idx = None
    end_idx = None
    
    # Make sure df exists and is not empty
    if df is None or df.empty:
        return None, None
    
    if selected_range and 'start_idx' in selected_range and 'end_idx' in selected_range:
        if selected_range['start_idx'] is not None and selected_range['end_idx'] is not None:
            # Validate indices to ensure they're within the range of the dataframe
            start_candidate = min(selected_range['start_idx'], selected_range['end_idx'])
            end_candidate = max(selected_range['start_idx'], selected_range['end_idx'])
            
            # Ensure indices are valid
            if start_candidate < 0:
                start_candidate = 0
            if end_candidate >= len(df):
                end_candidate = len(df) - 1
                
            # Only set if we have valid indices
            if start_candidate <= end_candidate and start_candidate < len(df) and end_candidate >= 0:
                start_idx = start_candidate
                end_idx = end_candidate
    
    # Split the data into selected and non-selected parts
    if start_idx is not None and end_idx is not None:
        # Data before selection
        before_df = df.iloc[:start_idx] if start_idx > 0 else None
        # Selected data
        selected_df = df.iloc[start_idx:end_idx+1]
        # Data after selection
        after_df = df.iloc[end_idx+1:] if end_idx < len(df)-1 else None
        
        # Plot data before selection
        if before_df is not None and not before_df.empty:
            fig.add_trace(
                go.Candlestick(
                    x=before_df['datetime'],
                    open=before_df['open'],
                    high=before_df['high'],
                    low=before_df['low'],
                    close=before_df['close'],
                    name="BTC Price (Before)",
                    increasing=dict(
                        line=dict(color=style['increasing_line_color']),
                        fillcolor=style['increasing_color']
                    ),
                    decreasing=dict(
                        line=dict(color=style['decreasing_line_color'], width=1.5),  # Thicker white outline
                        fillcolor=style['decreasing_color']
                    ),
                    showlegend=False,
                    hoverinfo="x+y"
                ),
                row=1, col=1
            )
        
        # Plot selected data with a different color/emphasis
        fig.add_trace(
            go.Candlestick(
                x=selected_df['datetime'],
                open=selected_df['open'],
                high=selected_df['high'],
                low=selected_df['low'],
                close=selected_df['close'],
                name=None,  # Removed "Selected Pattern" title
                increasing=dict(
                    line=dict(color='#4287f5', width=2),  # Light blue instead of green
                    fillcolor='rgba(66, 135, 245, 0.7)'  # Light blue with transparency
                ),
                decreasing=dict(
                    line=dict(color='#1a56c4', width=2),  # Darker blue instead of red
                    fillcolor='rgba(26, 86, 196, 0.7)'   # Darker blue with transparency
                ),
                hoverinfo="x+y"
            ),
            row=1, col=1
        )
        
        # Plot data after selection
        if after_df is not None and not after_df.empty:
            fig.add_trace(
                go.Candlestick(
                    x=after_df['datetime'],
                    open=after_df['open'],
                    high=after_df['high'],
                    low=after_df['low'],
                    close=after_df['close'],
                    name="BTC Price (After)",
                    increasing=dict(
                        line=dict(color=style['increasing_line_color']),
                        fillcolor=style['increasing_color']
                    ),
                    decreasing=dict(
                        line=dict(color=style['decreasing_line_color'], width=1.5),  # Thicker white outline
                        fillcolor=style['decreasing_color']
                    ),
                    showlegend=False,
                    hoverinfo="x+y"
                ),
                row=1, col=1
            )
        
        # Volume charts removed as requested
        
        # Add vertical lines to mark selection boundaries
        start_date = df.iloc[start_idx]['datetime']
        end_date = df.iloc[end_idx]['datetime']
        
        # Add vertical lines to mark the selection boundaries
        # Start line
        fig.add_shape(
            type="line",
            x0=start_date,
            x1=start_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#4287f5", width=2, dash="dash"),  # Light blue instead of yellow
            row=1, col=1
        )
        
        # End line
        fig.add_shape(
            type="line",
            x0=end_date,
            x1=end_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#4287f5", width=2, dash="dash"),  # Light blue instead of yellow
            row=1, col=1
        )
        
        # Add text annotations
        fig.add_annotation(
            x=start_date,
            y=1,
            text="START",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#4287f5",  # Light blue instead of yellow
            yref="paper",
            row=1, col=1
        )
        
        fig.add_annotation(
            x=end_date,
            y=1,
            text="END",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#4287f5",  # Light blue instead of yellow
            yref="paper",
            row=1, col=1
        )
    else:
        # If no selection, just show the regular chart
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="BTC Price",
                increasing=dict(
                    line=dict(color=style['increasing_line_color']),
                    fillcolor=style['increasing_color']
                ),
                decreasing=dict(
                    line=dict(color=style['decreasing_line_color'], width=1.5),  # Thicker white outline
                    fillcolor=style['decreasing_color']
                ),
                hoverinfo="x+y"
            ),
            row=1, col=1
        )
        
        # Volume chart removed as requested
    
    # Layout configuration with adjusted height (25% taller)
    fig.update_layout(
        height=690,  # Increased by 25% from 550 to ~690
        xaxis_rangeslider_visible=False,
        plot_bgcolor=style['background_color'],
        paper_bgcolor=style['background_color'],
        font=dict(color='white', family="ProtoMono-Light, monospace"),
        margin=dict(l=25, r=25, t=15, b=15),  # Reduced top/bottom margins as well
        hovermode='x unified',
        showlegend=False,  # Remove "trace 1" legend
        hoverlabel=dict(
            bgcolor=style['background_color'],
            font_size=14,
            font_family="ProtoMono-Light, monospace"
        ),
        # Remove all borders
        shapes=[],      # No shapes by default
        annotations=[], # No annotations by default
        xaxis_showticklabels=True,
        yaxis_showticklabels=True,
        xaxis_showspikes=False,
        yaxis_showspikes=False,
        modebar_remove=["lasso", "select"]
    )
    
    # Update x-axes with grid lines and flat time labels
    # Adjust tick spacing based on chart history length
    days_to_display = st.session_state.days_to_load if 'days_to_load' in st.session_state else 30
    
    # Determine appropriate tick spacing based on chart history length
    if days_to_display <= 30:  # Up to 1 month
        tick_spacing = "D1"  # Every day
        nticks = 30  # More ticks for shorter timeframes
    elif days_to_display <= 90:  # 1-3 months
        tick_spacing = "D5"  # Every 5 days
        nticks = 18  # ~90/5 = 18 ticks
    elif days_to_display <= 365:  # 3 months to 1 year
        tick_spacing = "D7"  # Every week
        nticks = 12  # ~12 weeks in 3 months
    else:  # More than a year
        tick_spacing = "D14"  # Every two weeks
        nticks = 26  # ~26 two-week periods in a year
    
    # Add 2.5% padding to the x-axis range to prevent candles from being cut off
    first_date = df.iloc[0]['datetime']
    last_date = df.iloc[-1]['datetime']
    date_range = last_date - first_date
    padding = date_range * 0.025  # 2.5% padding
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor=style['grid_color'],
        gridwidth=0.5,
        zeroline=False,
        showticklabels=True,
        linecolor=style['grid_color'],
        tickangle=0,  # Flat time labels
        tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),  # Consistent grey color
        tickformat="%m-%d<br>%H:%M",  # Two-line format with month-day on top, hours below
        dtick=tick_spacing,  # Dynamic tick spacing based on chart history
        tick0="2000-01-01",  # Start counting from a fixed date
        tickmode="auto",  # Change to auto mode to respect nticks
        nticks=nticks,  # Control the number of ticks dynamically based on date range
        ticks="outside",  # Place ticks outside the chart
        ticklen=8,  # Longer tick marks
        minor_showgrid=True,  # Show minor grid lines too
        minor_gridcolor=style['grid_color'],  # Ensure minor gridlines match color
        tickcolor="#999999",  # Ensure tick marks are grey
        range=[first_date - padding, last_date + padding],  # Add 5% padding on both sides
        showline=False,  # No border line on x-axis
        mirror=False,    # No mirrored border
        tickformatstops=[
            dict(dtickrange=[None, 86400000], value="%m-%d<br>%H:%M"),  # For daily view
            dict(dtickrange=[86400000, "M1"], value="%m-%d"),  # For monthly view
            dict(dtickrange=["M1", None], value="%Y-%m")  # For yearly view
        ]
    )
    
    # Set y-axis ranges based on data to prevent y-axis starting at zero
    # For price chart - use data range with some padding
    price_min = df['low'].min() * 0.98  # 2% padding below (increased for better visualization)
    price_max = df['high'].max() * 1.02  # 2% padding above (increased for better visualization)
    
    # Update y-axis with appropriate range and add grid
    # Set different tick spacing based on selected coin
    if st.session_state.selected_coin == "ETH":
        # ETH uses smaller price increments
        tick_spacing = 50    # Grid line every 50 price level for ETH
        tickformatstops_config = [
            dict(dtickrange=[None, 250], value=",.0f"),   # Show every 50 tick label
            dict(dtickrange=[250, None], value=",.0f")    # Show only every 250 tick label
        ]
    else:
        # BTC uses larger price increments
        tick_spacing = 1000  # Grid line every 1k price level for BTC
        tickformatstops_config = [
            dict(dtickrange=[None, 5000], value=",.0f"),  # Show every 1k tick label
            dict(dtickrange=[5000, None], value=",.0f")   # Show only every 5k tick label
        ]
    
    fig.update_yaxes(
        showgrid=True,  # Add horizontal grid lines
        gridcolor=style['grid_color'],
        gridwidth=0.5,
        zeroline=False,
        linecolor=style['grid_color'],
        range=[price_min, price_max],
        dtick=tick_spacing,  # Use dynamic tick spacing based on coin
        tickmode="linear",   # Use linear tick mode
        tick0=0,             # Start from 0
        tickformat=",.0f",   # No decimal places, comma for thousands
        tickformatstops=tickformatstops_config,  # Use dynamic tick format stops based on coin
        tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),  # Consistent grey color
        tickcolor="#999999",  # Ensure tick marks are grey
        hoverformat=",.0f",  # Format y-axis hover values as integers with no decimal places
        showline=False,      # No border line on y-axis
        mirror=False,        # No mirrored border
        row=1, col=1
    )

    # Config for more interactivity
    config = {
        'scrollZoom': True,
        'displaylogo': False,
    }
    
    return fig, config

# Function to find similar patterns
async def find_similar_patterns(
    symbol="BTC",     # Default to BTC
    interval="30m",   # Timeframe
    start_time=None,  # OPTIONAL: Start timestamp (used if not providing start_idx)
    end_time=None,    # OPTIONAL: End timestamp (used if not providing end_idx)
    start_idx=None,   # OPTIONAL: Start index in dataframe (preferred over timestamp)
    end_idx=None,     # OPTIONAL: End index in dataframe (preferred over timestamp)
    df=None,          # OPTIONAL: Dataframe for index lookup (required if using indices)
    max_matches=5,    # Maximum number of matches to return
    following_points=None,  # Will be dynamically calculated based on pattern length
    search_start=None,      # Optional search range start
    search_end=None,        # Optional search range end
    source_idx_range=None   # Add parameter for source index range - DEPRECATED
):
    """Find similar patterns using the Matrix Profile approach"""
    pattern_finder = get_pattern_finder()
    
    # Handle new index-based selection (preferred method) or timestamp-based selection
    if start_idx is not None and end_idx is not None and df is not None:
        # Use index-based selection - this ensures we get the exact number of candles
        # Extract the pattern from the dataframe using indices
        source_pattern = df.iloc[start_idx:end_idx+1][["timestamp", "open", "high", "low", "close", "volume"]]
        
        # Convert to dict format expected by pattern finder
        source_pattern_dict = source_pattern.to_dict("records")
        
        # Get the actual timestamps for search range based on indices
        start_time = df.iloc[start_idx]['datetime']
        end_time = df.iloc[end_idx]['datetime']
        
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
    else:
        # Fall back to timestamp-based selection (legacy method)
        # Format dates if needed
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
        
        # No source pattern available - will be extracted via date range
        source_pattern_dict = None
    
    # Format search range dates if provided
    search_start_iso = None
    if search_start is not None and isinstance(search_start, datetime):
        search_start_iso = search_start.isoformat()
    
    search_end_iso = None
    if search_end is not None and isinstance(search_end, datetime):
        search_end_iso = search_end.isoformat()
    
    # Run the pattern search 
    if source_pattern_dict is not None:
        # Use the new source_pattern_dict parameter instead of timestamps
        results = await pattern_finder.find_similar_patterns(
            symbol=symbol,  # Pass the symbol to use
            interval=interval,
            source_pattern=source_pattern_dict,  # Pass explicit pattern data instead of timestamps
            max_matches=max_matches,
            following_points=following_points,
            search_start_time=search_start_iso,
            search_end_time=search_end_iso
        )
    else:
        # Fall back to timestamp-based method (legacy)
        results = await pattern_finder.find_similar_patterns(
            symbol=symbol,  # Pass the symbol to use
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            max_matches=max_matches,
            following_points=following_points,
            search_start_time=search_start_iso,
            search_end_time=search_end_iso
        )
    
    return results


# Function to find similar patterns using feature extraction
async def find_similar_patterns_feature_based(
    symbol="BTC",     # Default to BTC
    interval="30m",   # Timeframe
    start_time=None,  # OPTIONAL: Start timestamp (used if not providing start_idx)
    end_time=None,    # OPTIONAL: End timestamp (used if not providing end_idx)
    start_idx=None,   # OPTIONAL: Start index in dataframe (preferred over timestamp)
    end_idx=None,     # OPTIONAL: End index in dataframe (preferred over timestamp)
    df=None,          # OPTIONAL: Dataframe for index lookup (required if using indices)
    max_matches=5,    # Maximum number of matches to return
    following_points=None,  # Will be dynamically calculated based on pattern length
    search_start=None,      # Optional search range start
    search_end=None,        # Optional search range end
    n_components=2          # Number of components for dimensionality reduction
):
    """Find similar patterns using the Feature Extraction approach"""
    pattern_finder = get_pattern_finder()
    
    # Handle new index-based selection (preferred method) or timestamp-based selection
    if start_idx is not None and end_idx is not None and df is not None:
        # Use index-based selection - this ensures we get the exact number of candles
        # Extract the pattern from the dataframe using indices
        source_pattern = df.iloc[start_idx:end_idx+1][["timestamp", "open", "high", "low", "close", "volume"]]
        
        # Convert to dict format expected by pattern finder
        source_pattern_dict = source_pattern.to_dict("records")
        
        # Get the actual timestamps for search range based on indices
        start_time = df.iloc[start_idx]['datetime']
        end_time = df.iloc[end_idx]['datetime']
        
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
    else:
        # Fall back to timestamp-based selection (legacy method)
        # Format dates if needed
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
        
        # No source pattern available - will be extracted via date range
        source_pattern_dict = None
    
    # Format search range dates if provided
    search_start_iso = None
    if search_start is not None and isinstance(search_start, datetime):
        search_start_iso = search_start.isoformat()
    
    search_end_iso = None
    if search_end is not None and isinstance(search_end, datetime):
        search_end_iso = search_end.isoformat()
    
    # Run the pattern search with feature extraction
    if source_pattern_dict is not None:
        # Use the source_pattern_dict parameter for explicit pattern data
        results = await pattern_finder.find_similar_patterns_feature_based(
            symbol=symbol,
            interval=interval,
            source_pattern=source_pattern_dict,
            max_matches=max_matches,
            following_points=following_points,
            search_start_time=search_start_iso,
            search_end_time=search_end_iso,
            n_components=n_components
        )
    else:
        # Fall back to timestamp-based method (legacy)
        results = await pattern_finder.find_similar_patterns_feature_based(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            max_matches=max_matches,
            following_points=following_points,
            search_start_time=search_start_iso,
            search_end_time=search_end_iso,
            n_components=n_components
        )
    
    return results

# Plot search results with Plotly using candlestick charts
def plot_match_results(source_pattern, matches, following_points=20, style=None, max_matches=10):
    if style is None:
        style = st.session_state.candle_style
    
    if not matches or len(matches) == 0:
        return None
    
    # Use up to max_matches
    num_matches = min(len(matches), max_matches)
    
    # Show more results in main view
    num_matches = min(num_matches, 10)  # Show up to 10 matches with consistent height
    
    # Create figure with 1+num_matches rows (source pattern + matches)
    fig = make_subplots(
        rows=1+num_matches, 
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.02,  # Minimal spacing between charts
        subplot_titles=None,  # Remove subplot titles entirely
        row_heights=[1] * (1+num_matches)  # All charts equal height
    )
    
    # Source pattern as candlestick
    source_df = pd.DataFrame(source_pattern)
    source_times = [datetime.fromtimestamp(ts/1000) for ts in source_df["timestamp"]]
    
    fig.add_trace(
        go.Candlestick(
            x=source_times,
            open=source_df['open'],
            high=source_df['high'],
            low=source_df['low'],
            close=source_df['close'],
            name=None,  # Removed "Source Pattern" title
            increasing=dict(
                line=dict(color=style['increasing_line_color']),
                fillcolor=style['increasing_color']
            ),
            decreasing=dict(
                line=dict(color=style['decreasing_line_color']),
                fillcolor=style['decreasing_color']
            ),
            hoverinfo="x+y"
        ),
        row=1, col=1
    )
    
    # Add source volume bars below the candlesticks
    source_volume_colors = ['rgba(0, 150, 136, 0.4)' if source_df['close'].iloc[i] >= source_df['open'].iloc[i] 
                        else 'rgba(239, 83, 80, 0.4)' 
                        for i in range(len(source_df))]
    
    # Add each match as candlestick charts
    for i in range(num_matches):
        match = matches[i]
        match_data = pd.DataFrame(match["pattern_data"])
        match_times = [datetime.fromtimestamp(ts/1000) for ts in match_data["timestamp"]]
        
        pattern_length = len(source_df)
        
        # Determine where to split - pattern part vs future part
        if len(match_times) > pattern_length:
            # Split the match dataframe
            pattern_part = match_data.iloc[:pattern_length]
            future_part = match_data.iloc[pattern_length-1:]  # Overlap by 1 candle
            
            pattern_times = match_times[:pattern_length]
            future_times = match_times[pattern_length-1:]
            
            # Add the pattern match part with blue theme
            fig.add_trace(
                go.Candlestick(
                    x=pattern_times,
                    open=pattern_part['open'],
                    high=pattern_part['high'],
                    low=pattern_part['low'],
                    close=pattern_part['close'],
                    name=None,  # Removed "Pattern Match" title
                    increasing=dict(
                        line=dict(color='rgba(66, 135, 245, 0.7)'),  # Light blue
                        fillcolor='rgba(66, 135, 245, 0.5)'          # Light blue with transparency
                    ),
                    decreasing=dict(
                        line=dict(color='rgba(26, 86, 196, 0.7)'),   # Darker blue
                        fillcolor='rgba(26, 86, 196, 0.5)'           # Darker blue with transparency
                    ),
                    hoverinfo="x+y"
                ),
                row=i+2, col=1
            )
            
            # Add the future part with normal colors
            fig.add_trace(
                go.Candlestick(
                    x=future_times,
                    open=future_part['open'],
                    high=future_part['high'],
                    low=future_part['low'],
                    close=future_part['close'],
                    name=None,  # Removed "What Happened After" title
                    increasing=dict(
                        line=dict(color=st.session_state.candle_style['increasing_line_color']),
                        fillcolor=st.session_state.candle_style['increasing_color']
                    ),
                    decreasing=dict(
                        line=dict(color=st.session_state.candle_style['decreasing_line_color']),
                        fillcolor=st.session_state.candle_style['decreasing_color']
                    ),
                    hoverinfo="x+y"
                ),
                row=i+2, col=1
            )
            
            # Add vertical line to separate pattern from future
            prediction_start = match_times[pattern_length-1]
            fig.add_shape(
                type="line",
                x0=prediction_start,
                x1=prediction_start,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="#4287f5", width=2, dash="dash"),  # Light blue instead of yellow
                row=i+2, col=1
            )
            
            # Add annotation
            fig.add_annotation(
                x=prediction_start,
                y=1,
                text="Pattern End / Future Start",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#4287f5",  # Light blue instead of yellow
                yref="paper",
                row=i+2, col=1
            )
        else:
            # Just show the whole match without splitting
            fig.add_trace(
                go.Candlestick(
                    x=match_times,
                    open=match_data['open'],
                    high=match_data['high'],
                    low=match_data['low'],
                    close=match_data['close'],
                    name=None,  # Removed match title
                    increasing=dict(
                        line=dict(color=style['increasing_line_color']),
                        fillcolor=style['increasing_color']
                    ),
                    decreasing=dict(
                        line=dict(color=style['decreasing_line_color']),
                        fillcolor=style['decreasing_color']
                    ),
                    hoverinfo="x+y"
                ),
                row=i+2, col=1
            )
            
            # Add padding to ensure first and last candles are fully visible
            if match_times:
                pattern_timespan = match_times[-1] - match_times[0]
                padding_time = pattern_timespan * 0.05
                view_start = match_times[0] - padding_time
                view_end = match_times[-1] + padding_time
    
    # Set all charts to the same height for consistency
    chart_height = 400  # Height per chart (same as original source pattern)
    title_gap = 30     # Additional height for titles
    
    # Total plot height includes all charts with equal height plus title space
    plot_height = (chart_height * (1 + num_matches)) + (title_gap * (1 + num_matches))
    
    fig.update_layout(
        height=plot_height,
        plot_bgcolor=style['background_color'],
        paper_bgcolor=style['background_color'],
        font=dict(color='white'),
        margin=dict(l=15, r=15, t=50, b=10),  # Reduced margins by 50%
        showlegend=False,  # Remove legends completely
        hoverlabel=dict(
            bgcolor=style['background_color'],
            font_size=14,
            font_family="ProtoMono-Light, monospace"
        ),
        hovermode='x unified'
    )
    
    # Update x-axes with grid lines and flat time labels - for all subplots
    fig.update_xaxes(
        showgrid=True,
        gridcolor=style['grid_color'],
        gridwidth=0.5,
        zeroline=False,
        linecolor=style['grid_color'],
        rangeslider_visible=False,
        tickangle=0,  # Flat time labels
        tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),  # Consistent grey color
        tickformat="%m-%d<br>%H:%M",  # Two-line format with month-day on top, hours below
        nticks=15,  # Increase number of x-axis divisions
        ticks="outside",  # Place ticks outside the chart
        ticklen=8,  # Longer tick marks
        minor_showgrid=True,  # Show minor grid lines too
        minor_gridcolor=style['grid_color'],  # Ensure minor gridlines match color
        tickcolor="#999999"  # Ensure tick marks are grey
    )
    
    # Auto-adjust y-axis for each subplot separately
    for i in range(1, num_matches + 2):  # +2 because row indices start at 1 and we have num_matches+1 rows
        # Get the data for this subplot
        if i == 1:
            # Source pattern
            subplot_data = source_df
        else:
            # Match data
            match_data = pd.DataFrame(matches[i-2]["pattern_data"])
            subplot_data = match_data
        
        # Calculate y-axis range
        if not subplot_data.empty:
            y_min = subplot_data['low'].min() * 0.995
            y_max = subplot_data['high'].max() * 1.005
            
            # Set different tick spacing based on selected coin
            if st.session_state.selected_coin == "ETH":
                # ETH uses smaller price increments
                tick_spacing = 50    # Grid line every 50 price level for ETH
                tickformatstops_config = [
                    dict(dtickrange=[None, 250], value=",.0f"),   # Show every 50 tick label
                    dict(dtickrange=[250, None], value=",.0f")    # Show only every 250 tick label
                ]
            else:
                # BTC uses larger price increments
                tick_spacing = 1000  # Grid line every 1k price level for BTC
                tickformatstops_config = [
                    dict(dtickrange=[None, 5000], value=",.0f"),  # Show every 1k tick label
                    dict(dtickrange=[5000, None], value=",.0f")   # Show only every 5k tick label
                ]
                
            # Update y-axis range with consistent styling
            fig.update_yaxes(
                showgrid=True,
                gridcolor=style['grid_color'],
                gridwidth=0.5,
                zeroline=False,
                linecolor=style['grid_color'],
                range=[y_min, y_max],
                dtick=tick_spacing,  # Dynamic grid line spacing based on coin
                tickmode="linear",  # Use linear tick mode
                tick0=0,  # Start from 0
                tickformat=",.0f",  # No decimal places, comma for thousands
                tickformatstops=tickformatstops_config,  # Dynamic tick format stops based on coin
                tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),  # Consistent grey color
                tickcolor="#999999",  # Ensure tick marks are grey
                hoverformat=",.0f",  # Format y-axis hover values as integers with no decimal places
                row=i, col=1
            )
    
    return fig

# Custom CSS to help position the title and market selector
st.markdown(
    """
    <style>
    /* Custom styles for title row */
    .title-with-selector {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0;
        padding: 0;
    }
    .title-with-selector h1 {
        margin: 0;
        padding: 0;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title only, no market selector here
col_title = st.container()

# Title in column
with col_title:
    st.markdown(
        """
        <h1 style="margin: 0; padding: 0; font-size: 1rem; margin-bottom: 0.1rem;">
            Pattern Searcher: Time Series Analysis Using Matrix Profiles and High-Dimensional Features Extraction by 
            <a href="https://x.com/figure31_" target="_blank" style="color: #4287f5; text-decoration: none;">Figure31</a>
        </h1>
        """, 
        unsafe_allow_html=True
    )

# Add ultra-minimal spacing after title and pull up controls
st.markdown(
    """
    <style>
    /* Target the horizontal block with controls to pull it up */
    section[data-testid="stSidebar"] + div > div:nth-child(4) {
        margin-top: -15px !important;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Add the horizontal alignment fix for all columns and selectboxes
st.markdown(
    """
    <style>
    /* Make all columns consistent in width behavior */
    div[data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
    }
    
    /* Give each selectbox consistent width within its column */
    div[data-testid="stSelectbox"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Override the previous CSS that was targeting specific selectboxes */
    div[data-testid="stSelectbox"] {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Remove any existing margin/width constraints set earlier */
    div.stSelectbox {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Ensure the column itself expands fully */
    [data-testid="column"]:nth-child(4) {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Top bar for interval and search controls with zero margins
cols = st.columns([0.2, 0.2, 1.3, 0.2])

with cols[0]:
    # Timeframe selector with improved styling - more compact
    st.markdown('<div style="margin: 0; padding: 0; font-size: 0.9rem;"><strong>Timeframe</strong></div>', unsafe_allow_html=True)
    intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    interval = st.selectbox(
        "Timeframe",
        options=intervals,
        index=intervals.index(st.session_state.interval) if st.session_state.interval in intervals else 4,  # Default to 30m (index 4)
        label_visibility="collapsed"
    )
    
    # Check if interval changed
    if interval != st.session_state.interval:
        # First update all the session state
        st.session_state.interval = interval
        st.session_state.btc_data = None
        st.session_state.selected_range = {'start_idx': None, 'end_idx': None}
        st.session_state.search_results = None  # Clear previous search results
        
        # Clear all interval-specific state to ensure a fresh setup
        for key in list(st.session_state.keys()):
            if key.startswith('pattern_selection_slider_') or key.startswith('prev_slider_values_') or key.startswith('initial_range_set_'):
                del st.session_state[key]
                
        # Force a page reload to ensure everything is synchronized
        st.rerun()
        
with cols[1]:
    # Create date range options based on the selected interval - more compact
    st.markdown('<div style="margin: 0; padding: 0; font-size: 0.9rem;"><strong>Chart History</strong></div>', unsafe_allow_html=True)
    
    # Define date range options appropriate for each timeframe
    date_range_options = {
        "1m": [1, 2, 3, 5, 7, 10, 14],  # 1m: up to 14 days
        "3m": [1, 2, 3, 5, 7, 10, 14, 21],  # 3m: up to 3 weeks
        "5m": [1, 2, 3, 5, 7, 10, 14, 21, 30],  # 5m: up to 1 month
        "15m": [1, 3, 7, 14, 21, 30, 60, 90],  # 15m: up to 3 months
        "30m": [1, 3, 7, 14, 30, 60, 90, 120, 180],  # 30m: up to 6 months
        "1h": [1, 3, 7, 14, 30, 60, 90, 180, 365],  # 1h: up to 1 year
        "4h": [7, 14, 30, 60, 90, 180, 365, 548, 730],  # 4h: up to 2 years
        "1d": [14, 30, 90, 180, 365, 548, 730, 1095],  # 1d: up to 3 years
        "1w": [30, 90, 180, 365, 730, 1095, 1825]  # 1w: up to 5 years
    }
    
    # Get appropriate options for the current interval
    current_options = date_range_options.get(interval, [30])
    
    # Create more user-friendly labels for days
    def format_days(days):
        if days == 1:
            return "1 day"
        elif days == 7:
            return "1 week"
        elif days == 14:
            return "2 weeks"
        elif days == 21:
            return "3 weeks"
        elif days == 30:
            return "1 month"
        elif days == 60:
            return "2 months"
        elif days == 90:
            return "3 months"
        elif days == 120:
            return "4 months"
        elif days == 180:
            return "6 months"
        elif days == 365:
            return "1 year"
        elif days == 548:
            return "1.5 years"
        elif days == 730:
            return "2 years"
        elif days == 1095:
            return "3 years"
        elif days == 1825:
            return "5 years"
        else:
            return f"{days} days"
    
    # Create list of formatted options
    display_options = [format_days(d) for d in current_options]
    
    # Initialize the session state if needed
    if 'days_to_load' not in st.session_state:
        # Default to 1 week (7 days)
        if 7 in current_options:
            default_index = current_options.index(7)  # Find the 7-day option
        else:
            # If 7 days is not available for this timeframe, pick closest available
            default_index = min(range(len(current_options)), key=lambda i: abs(current_options[i] - 7))
        
        st.session_state.days_to_load = current_options[default_index]
        st.session_state.days_display = display_options[default_index]
    
    # Handle the case where switching intervals might invalidate the current selection
    if interval != st.session_state.interval and 'days_to_load' in st.session_state:
        if st.session_state.days_to_load not in current_options:
            # Prefer 7 days (1 week) or closest available option
            if 7 in current_options:
                default_index = current_options.index(7)
            else:
                # Pick closest available to 7 days
                default_index = min(range(len(current_options)), key=lambda i: abs(current_options[i] - 7))
                
            st.session_state.days_to_load = current_options[default_index]
            st.session_state.days_display = display_options[default_index]
    
    # Create the selectbox for date range
    selected_range = st.selectbox(
        "History Range",
        options=display_options,
        index=current_options.index(st.session_state.days_to_load) if st.session_state.days_to_load in current_options else 0,
        label_visibility="collapsed"
    )
    
    # Update session state with the selected date range
    selected_index = display_options.index(selected_range)
    days = current_options[selected_index]
    
    # Only reload data if the days value changed
    if days != st.session_state.days_to_load:
        st.session_state.days_to_load = days
        st.session_state.days_display = selected_range
        st.session_state.btc_data = None  # Trigger data reload
        st.session_state.search_results = None  # Clear previous search results
        
with cols[3]:
    # Initialize session state for selected_coin if not present
    if 'selected_coin' not in st.session_state:
        st.session_state.selected_coin = "BTC"
    
    # Use the same label style that's used for the other columns
    st.markdown('<div style="margin: 0; padding: 0; font-size: 0.9rem;"><strong>Market</strong></div>', unsafe_allow_html=True)
    
    # Create selectbox with no custom styling - exactly like other selectboxes
    selected_coin = st.selectbox(
        "Market",
        options=["BTC", "ETH"],
        index=0 if st.session_state.selected_coin == "BTC" else 1,
        key="coin_selector",
        label_visibility="collapsed"
    )
    
    # Update session state if changed
    if selected_coin != st.session_state.selected_coin:
        st.session_state.selected_coin = selected_coin
        # Reset data when changing coins
        st.session_state.btc_data = None
        st.session_state.selected_range = {'start_idx': None, 'end_idx': None}
        st.session_state.search_results = None
        st.rerun()

# Remove duplicate col2 definition
# Don't overwrite candle style - it's already set in session state initialization
# The hollow candle style is already defined

with cols[2]:
    # Search period title - more compact
    st.markdown('<div style="margin: 0; padding: 0; font-size: 0.9rem;"><strong>Search Period</strong></div>', unsafe_allow_html=True)
    # Create date pickers for more precise date selection
    # Set default start date to 6 months before the last candle on the main chart
    if 'search_start_date' not in st.session_state or 'btc_data' in st.session_state and st.session_state.btc_data is not None:
        # If we have data, calculate from last candle
        if 'btc_data' in st.session_state and st.session_state.btc_data is not None:
            df = st.session_state.btc_data
            if not df.empty:
                last_candle_date = df.iloc[-1]['datetime']
                st.session_state.search_start_date = last_candle_date - timedelta(days=180)  # 6 months
            else:
                st.session_state.search_start_date = datetime(MIN_YEAR, 1, 1)
        else:
            st.session_state.search_start_date = datetime(MIN_YEAR, 1, 1)
            
    if 'search_end_date' not in st.session_state:
        st.session_state.search_end_date = datetime.now()
    
    # Pre-calculate values
    years = list(range(MIN_YEAR, datetime.now().year + 1))
    months = ["January", "February", "March", "April", "May", "June", 
             "July", "August", "September", "October", "November", "December"]
    month_options = [f"{i}: {months[i-1]}" for i in range(1, 13)]
    
    # Single-line date range selector with compact sizes
    date_cols = st.columns([2.5, 2.5, 2.5, 0.01, 2.5, 2.5, 2.5])
    
    # Get the date of the latest candle for end date
    latest_candle_date = datetime.now()  # Default to today
    if 'btc_data' in st.session_state and st.session_state.btc_data is not None:
        df = st.session_state.btc_data
        if not df.empty:
            latest_candle_date = df.iloc[-1]['datetime']
    
    # Calculate default start date (6 months before end date)
    default_start_date = latest_candle_date - timedelta(days=180)  # ~6 months
    
    # Make sure default start date isn't before MIN_YEAR
    if default_start_date.year < MIN_YEAR:
        default_start_date = datetime(MIN_YEAR, 1, 1)
    
    # Start date components
    # Year selection
    start_year_index = years.index(default_start_date.year) if default_start_date.year in years else 0
    
    with date_cols[0]:
        start_year = st.selectbox(
            "Year",
            options=years,
            index=start_year_index,
            key="start_year",
            label_visibility="collapsed"
        )
    
    # Month selection
    start_month_index = default_start_date.month - 1  # Adjust for 0-indexed list
    
    with date_cols[1]:
        month_index = st.selectbox(
            "Month",
            options=month_options,
            index=start_month_index,
            key="start_month",
            label_visibility="collapsed"
        )
        start_month = int(month_index.split(":")[0])
    
    # Day selection
    import calendar
    _, max_days = calendar.monthrange(start_year, start_month)
    default_start_day = min(default_start_date.day, max_days)
    start_day_index = default_start_day - 1  # Adjust for 0-indexed list
    
    with date_cols[2]:
        start_day = st.selectbox(
            "Day",
            options=range(1, max_days + 1),
            index=start_day_index,
            key="start_day",
            label_visibility="collapsed"
        )
    
    # End date spacer without "To:" text
    with date_cols[3]:
        st.write(" ")
    
    # Set end date defaults from the latest candle date
    latest_year = latest_candle_date.year
    latest_month = latest_candle_date.month
    latest_day = latest_candle_date.day
    
    # Year selection for end date
    with date_cols[4]:
        # Find index of the latest year
        latest_year_index = years.index(latest_year) if latest_year in years else len(years) - 1
        end_year = st.selectbox(
            "Year",
            options=years,
            index=latest_year_index,
            key="end_year",
            label_visibility="collapsed"
        )
    
    # Month selection for end date
    with date_cols[5]:
        latest_month_index = latest_month - 1  # Adjust for 0-based index
        month_index = st.selectbox(
            "Month",
            options=month_options,
            index=latest_month_index,
            key="end_month",
            label_visibility="collapsed"
        )
        end_month = int(month_index.split(":")[0])
    
    # Day selection for end date
    with date_cols[6]:
        _, max_days = calendar.monthrange(end_year, end_month)
        # Find the right day index, but don't exceed max days in the month
        latest_day_index = min(latest_day, max_days) - 1  # Adjust for 0-based index
        end_day = st.selectbox(
            "Day",
            options=range(1, max_days + 1),
            index=latest_day_index,
            key="end_day",
            label_visibility="collapsed"
        )
    
    # Create datetime objects
    try:
        search_start_date = datetime(start_year, start_month, start_day)
        search_end_date = datetime(end_year, end_month, end_day, 23, 59, 59)
        
        # Validate date range
        if search_end_date < search_start_date:
            st.error("End date must be after start date")
            search_end_date = search_start_date + timedelta(days=1)
    except ValueError as e:
        st.error(f"Invalid date: {str(e)}")
        search_start_date = datetime(MIN_YEAR, 1, 1)
        search_end_date = datetime.now()
    
    # Update session state
    st.session_state.search_range = {
        'start_date': search_start_date,
        'end_date': search_end_date
    }

# Load data if needed - with progress information
if st.session_state.btc_data is None:
    with st.spinner(f"Loading {st.session_state.selected_coin} data..."):
        progress_bar = st.progress(0)
        
        # Use the user-selected days value from session state
        if 'days_to_load' in st.session_state:
            days_to_load = st.session_state.days_to_load
            days_display = st.session_state.days_display
        else:
            # Fallback to default values if session state is not yet initialized
            interval_to_days = {
                "1m": 3,     # Very short timeframe
                "3m": 5,     # Short timeframe
                "5m": 7,     # Short timeframe
                "15m": 14,   # Medium-short timeframe
                "30m": 30,   # Medium timeframe
                "1h": 60,    # Medium-long timeframe
                "4h": 120,   # Long timeframe
                "1d": 500,   # Very long timeframe
                "1w": 1500   # Extremely long timeframe
            }
            days_to_load = interval_to_days.get(interval, 30)
            days_display = f"{days_to_load} days"
        
        progress_bar.progress(10, text=f"Loading {days_to_load} days of {interval} {st.session_state.selected_coin} data...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(load_market_data(
            symbol=st.session_state.selected_coin,
            interval=interval, 
            days=days_to_load
        ))
        
        if data.empty:
            progress_bar.progress(100, text="Error: Could not fetch data from Binance API")
            time.sleep(1.5)  # Longer pause to show error
            progress_bar.empty()
            st.error(f"Could not fetch {st.session_state.selected_coin} data from Binance API. Please try again later.")
        else:
            st.session_state.btc_data = data
            progress_bar.progress(100, text="Data loaded successfully!")
            time.sleep(0.5)  # Short pause to show success
            progress_bar.empty()  # Remove progress bar

# Main content - Chart + Controls
if st.session_state.btc_data is not None:
    df = st.session_state.btc_data
    
    # Display the chart first and then selection controls below
    # This layout gives more prominence to the chart
    
    # ALWAYS reset selection for the current timeframe's first load or when the range is invalid
    initial_range_key = f"initial_range_set_{interval}"
    
    # Force a reset either when first loading a timeframe or when the range is invalid
    need_reset = (
        initial_range_key not in st.session_state or 
        not st.session_state[initial_range_key] or
        st.session_state.selected_range['start_idx'] is None or
        st.session_state.selected_range['start_idx'] >= len(df) or
        st.session_state.selected_range['end_idx'] >= len(df)
    )
    
    if need_reset:
        # Set default range - always use last 50 candles
        default_end = len(df) - 1
        default_start = max(0, default_end - 49)  # 50 candles (end - start + 1 = 50)
        
        # Initialize the selection with our defaults
        st.session_state.selected_range = {
            'start_idx': default_start,
            'end_idx': default_end
        }
        
        # Mark that we've set the initial range for this timeframe
        st.session_state[initial_range_key] = True
        
        # Debugging help
        print(f"Reset selection range for {interval}: {default_start} to {default_end}")
    else:
        # Use the existing selection range
        default_start = st.session_state.selected_range['start_idx']
        default_end = st.session_state.selected_range['end_idx']
    
    # Generate an interactive candlestick chart with current selection
    fig, config = create_candlestick_chart(df, st.session_state.selected_range)
    
    # Display chart with plotly - make it prominent with reduced margins
    st.markdown(
        """
        <style>
        /* Make chart more compact */
        div.element-container:has(div.stPlotlyChart) {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Display the number of candles prominently - without blue info box - even more compact
    if st.session_state.selected_range['start_idx'] is not None and st.session_state.selected_range['end_idx'] is not None:
        start_idx = st.session_state.selected_range['start_idx']
        end_idx = st.session_state.selected_range['end_idx']
        
        start_date = df.iloc[start_idx]['datetime'].strftime("%Y-%m-%d %H:%M")
        end_date = df.iloc[end_idx]['datetime'].strftime("%Y-%m-%d %H:%M")
        num_candles = end_idx - start_idx + 1
        
        # Use markdown with minimal spacing
        st.markdown(f"<div style='padding: 0; margin: 0; margin-top: -10px; margin-bottom: 5px; font-size: 0.9rem;'><strong>Selected Pattern:</strong> {start_date} to {end_date} ({num_candles} candles)</div>", unsafe_allow_html=True)
    
    # Use a single range slider to select both start and end with one control
    
    # Create a range slider with step of 1 for more granular control
    # Validate our default values to ensure they're in range
    if default_start >= len(df):
        default_start = max(0, len(df) - 1)
    if default_end >= len(df):
        default_end = len(df) - 1
    
    # Store previous slider values to detect changes - use timeframe-specific key
    prev_values_key = f"prev_slider_values_{interval}"
    if prev_values_key not in st.session_state:
        st.session_state[prev_values_key] = (default_start, default_end)
    
    # Apply custom styling to make the slider more compact
    st.markdown(
        """
        <style>
        /* Make the slider less tall */
        div[data-testid="stSlider"] {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 5px !important;
        }
        
        /* Make caption more compact */
        small {
            margin-top: 0 !important;
            padding-top: 0 !important;
            font-size: 0.7rem !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Create a unique key for the slider based on the timeframe to ensure full reset when switching
    slider_key = f"pattern_selection_slider_{interval}"
    
    # Create the slider that will update automatically
    start_slider, end_slider = st.select_slider(
        "Pattern Selection",  # Proper non-empty label
        options=list(range(len(df))),
        value=(default_start, default_end),
        label_visibility="collapsed",  # Hide the label but it's there for accessibility
        key=slider_key  # Use the timeframe-specific key
    )
    
    # Removed caption text as requested
    
    # Check if slider values have changed from previous values
    prev_values_key = f"prev_slider_values_{interval}"
    if (start_slider, end_slider) != st.session_state[prev_values_key]:
        # Update session state with new range
        st.session_state.selected_range = {
            'start_idx': start_slider,
            'end_idx': end_slider
        }
        # Store current values for next comparison
        st.session_state[prev_values_key] = (start_slider, end_slider)
        
        # Scroll position is now automatically managed by our scroll anchor
        # No need for additional JavaScript here
        
        # Rerun to update the chart
        st.rerun()
    
    # Add session state variables if they don't exist
    if 'max_matches' not in st.session_state:
        st.session_state.max_matches = 50  # Default to 50 matches
        
    if 'max_distance_pct' not in st.session_state:
        st.session_state.max_distance_pct = 150  # Default to 1.5x multiplier
    
    if 'n_components' not in st.session_state:
        st.session_state.n_components = 2  # Default to 2 components
    
    # Create columns for pattern search controls
    pattern_search_cols = st.columns([2, 1])
    
    # Add CSS to match the number inputs exactly to the other button styles in the app
    st.markdown(
        """
        <style>
        /* Style the button and number input containers to match them exactly */
        .stButton button,
        .stNumberInput [data-baseweb="input"],
        .stNumberInput [data-baseweb="base-input"],
        .stSelectbox [data-baseweb="select"] {
            background-color: #2c2c2c !important;
            color: #cccccc !important;
            border-color: #444444 !important;
            border-width: 1px !important;
            border-style: solid !important;
            border-radius: 4px !important;
        }
        
        /* Match hover states */
        .stButton button:hover,
        .stNumberInput:hover [data-baseweb="input"],
        .stNumberInput:hover [data-baseweb="base-input"],
        .stSelectbox:hover [data-baseweb="select"] {
            background-color: #3a3a3a !important;
            border-color: #555555 !important;
        }
        
        /* Ensure the text color matches exactly */
        .stButton button span,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] span {
            color: #cccccc !important;
        }
        
        /* Make labels match */
        .stNumberInput label,
        .stSelectbox label {
            color: #cccccc !important;
        }
        
        /* Completely remove spinner buttons */
        input[type="number"]::-webkit-inner-spin-button, 
        input[type="number"]::-webkit-outer-spin-button { 
            -webkit-appearance: none !important;
            margin: 0 !important;
            opacity: 0 !important;
        }
        
        /* Make sure the input itself is properly styled */
        .stNumberInput input {
            background-color: #2c2c2c !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Remove any artifacts or extras */
        .stNumberInput [data-baseweb="input"]::after,
        .stNumberInput [data-baseweb="input"]::before,
        .stNumberInput [data-baseweb="base-input"]::after,
        .stNumberInput [data-baseweb="base-input"]::before {
            display: none !important;
            content: none !important;
        }
        
        /* Hide any visual markers we don't want */
        [data-testid="stMarkdownContainer"] small {
            display: none !important;
        }
        
        /* Fix for temporary slider appearing near analysis method selector */
        div.element-container:has(div[data-baseweb="slider"]) ~ div.element-container:has(div[data-baseweb="slider"]) {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            position: absolute !important;
            pointer-events: none !important;
        }
        
        /* Ensure sliders adjacent to selectboxes don't create temporary UI elements */
        div.element-container:has(div[data-baseweb="select"]) ~ div.element-container:has(div[data-baseweb="slider"]) {
            opacity: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
            position: absolute !important;
            z-index: -999 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Create temporary variables for the current values
    current_max_matches = st.session_state.max_matches
    current_multiplier = st.session_state.max_distance_pct / 100.0
    current_multiplier = min(current_multiplier, 3.0)  # Ensure it's within range
    
    # Create sliders instead of number inputs - they style more consistently
    # Create an additional row for analysis method selection and components
    analysis_method_cols = st.columns([1, 1])
    
    with analysis_method_cols[0]:
        # Add analysis method selection
        st.markdown("**Analysis Method**")
        analysis_method = st.selectbox(
            "Analysis Method",
            options=["Matrix Profile", "Feature Extraction"],
            index=0 if "analysis_method" not in st.session_state else 
                  (0 if st.session_state.analysis_method == "Matrix Profile" else 1),
            label_visibility="collapsed",
            help="Matrix Profile finds exact shape matches. Feature Extraction finds patterns with similar statistical properties."
        )
        # Store the selected method in session state
        st.session_state.analysis_method = analysis_method
    
    with analysis_method_cols[1]:
        # Always create a container for consistency regardless of method
        components_container = st.container()
        
        # Show component selector only if Feature Extraction is selected
        if analysis_method == "Feature Extraction":
            with components_container:
                st.markdown("**Number of Components**")
                component_options = [2, 3, 4, 5]
                n_components = st.selectbox(
                    "Components",
                    options=component_options,
                    index=component_options.index(st.session_state.n_components) if hasattr(st.session_state, 'n_components') and st.session_state.n_components in component_options else 0,
                    label_visibility="collapsed",
                    key="components_selector",  # Add explicit key to avoid conflicts
                    help="Number of components for dimensionality reduction. 2-3 components visualize well, more components can capture more complex relationships."
                )
                # Store the components value
                st.session_state.n_components = n_components
        else:
            # When Matrix Profile is selected, create an empty placeholder to maintain layout
            with components_container:
                # Need to render something hidden to maintain layout
                st.markdown('<div style="display:none;">placeholder</div>', unsafe_allow_html=True)
    
    # Add spacing before guidance text - increase it to match what's below
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    # Add guidance text that spans the full width
    if analysis_method == "Matrix Profile":
        st.markdown("""
        **Matrix Profile - Find Visually Similar Patterns:** Matrix Profile finds visually similar patterns using a shape-based matching algorithm. It's ideal for finding historical occasions where price action behaved similarly to what you're seeing now or at a specific point in time. You can use it to confirm or refute specific chart formations you've identified. Setting tip: A multiplier filter setting of 1.2-1.5 provides strict matching for closer pattern similarity, while higher values up to 2.0 offers more flexible matching to capture broader pattern families. Score matches below 2 are ideal, above 4 the system is less reliable. For pattern selection, 15-50 candles generally works well, though up to 75 can still provide meaningful results. Beyond 100 candles, matches become increasingly rare and less precise. We recommend setting 50-100 maximum matches initially to explore similar historical periods. Important: Always examine the actual match charts rather than relying solely on the higher/lower statistics. Our median price comparison uses a fixed timeframe equal to 2 times the length of your selected pattern, which may not align with your specific trading horizon. The patterns themselves often reveal nuances and potential outcomes that statistics alone cannot capture. For trading guidance, if 70%+ of historical patterns moved in one direction, this may suggest a bias worth exploring further with your own analysis.
        """)
    else:  # feature_extraction
        st.markdown("""
        **Feature Extraction - Find Statistically Similar Patterns:** Feature Extraction uses Principal Component Analysis (PCA) to find statistically similar patterns based on underlying market conditions. This method excels at identifying market conditions and regime changes through statistical relationships—analyzing volatility clustering, momentum divergences, trend strength transitions, and complex intermarket correlations. It's more statistical than visual. The scatter plot shows pattern groupings where the blue dot represents your selected pattern, teal dots show matches, and proximity indicates similarity level. Patterns are matched using statistical indicators like volatility, trend strength, and candlestick characteristics rather than visual shape. Lower distance scores (under 2.0) indicate stronger statistical similarity, with scores below 1.0 representing particularly strong matches. Unlike visual matching, these scores represent distances in high-dimensional feature space (typically 15-25 dimensions reduced through PCA)—patterns with similar volatility signatures, trend momentum, and price action characteristics cluster together regardless of visual appearance. For component selection, start with 2 components (default). If explained variance is below 65%, try increasing to 3-4 components to capture more statistical detail, though higher values may include noise. Start with 50-100 maximum matches to identify the primary pattern clusters. For deeper regime analysis or system development, increase to 200+ matches to better understand the full feature landscape. Feature extraction works best with 20-40 candle patterns that capture sufficient market behavior to extract meaningful statistical properties. Important: Always examine the actual match charts rather than relying solely on the higher/lower statistics. Our median price comparison uses a fixed timeframe equal to 2 times the length of your selected pattern, which may not align with your specific trading horizon. The patterns themselves often reveal nuances and potential outcomes that statistics alone cannot capture. While these patterns may not look the same, they often lead to similar market behavior, making this method more sophisticated than the more straightforward matrix profile analysis.
        """)
        
    # Add spacing after guidance text - keep the same amount
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    with pattern_search_cols[0]:
        # Create a simple slider with a proper label
        st.markdown("**Maximum Matches**")
        max_matches = st.slider(
            "Maximum Matches",  # Provide a non-empty label to avoid warnings
            min_value=10,
            max_value=300,
            value=current_max_matches,
            step=5,
            label_visibility="collapsed",  # Hide the label since we show it with markdown
            help="Set the maximum number of patterns to find. Higher values will find more patterns but may take longer to process."
        )
    
    # For the second column, make a global decision whether to show multiplier or not
    # This needs to be before the column definition
    if analysis_method == "Matrix Profile":
        show_multiplier = True
    else:
        show_multiplier = False
        multiplier = current_multiplier  # Set default value when not showing slider
            
    with pattern_search_cols[1]:
        # Use a dummy container for both cases
        if show_multiplier:
            # Only create the UI elements for Matrix Profile
            st.markdown("<strong>Filter (score multiplier)</strong>", unsafe_allow_html=True)
            multiplier = st.slider(
                "Filter Score Multiplier",
                min_value=1.0,
                max_value=3.0,
                value=current_multiplier,
                step=0.1,
                format="%.1fx",
                label_visibility="collapsed",
                key="filter_slider",
                help="Filter out patterns with scores above this multiplier of the best match score. Lower values = more similar patterns only."
            )
        else:
            # For Feature Extraction - just create an empty placeholder to maintain layout
            # but don't create any UI controls that could appear temporarily
            st.empty()
        
    # Store the search parameters in session state to minimize UI rebuilding during search
    if 'search_started' not in st.session_state:
        st.session_state.search_started = False
    
    # Use session state to track if we need to start a search
    if st.button("Find Similar Patterns", type="secondary", use_container_width=True, key="search_button"):
        # Store all search parameters in session state to avoid UI rebuilding
        st.session_state.max_matches = max_matches
        st.session_state.max_distance_pct = int(multiplier * 100)
        st.session_state.search_started = True
        
        # Force a rerun to trigger the search code below
        st.rerun()
    
    # Run search in a separate code path to avoid UI rebuilding during search
    if st.session_state.search_started:
        # Reset the flag immediately
        st.session_state.search_started = False
        
        # Check if a pattern is selected
        if st.session_state.selected_range['start_idx'] is None or st.session_state.selected_range['end_idx'] is None:
            st.error("Please select a pattern on the chart first")
        else:
            # Show search progress 
            with st.spinner("Searching..."):
                progress_bar = st.progress(0)
                
                # Get datetime values
                start_idx = st.session_state.selected_range['start_idx']
                end_idx = st.session_state.selected_range['end_idx']
                start_time = df.iloc[start_idx]['datetime']
                end_time = df.iloc[end_idx]['datetime']
                
                progress_bar.progress(20)
                
                # Double-check the pattern length before searching
                pattern_length = end_idx - start_idx + 1
                if pattern_length < 4:
                    progress_bar.progress(100, text=f"Error: Pattern too short ({pattern_length} candles). Need at least 4 candles.")
                    time.sleep(1.5)
                    progress_bar.empty()
                    st.error(f"Pattern is too short. Please select at least 4 candles. (Selected: {pattern_length} candles)")
                else:
                    # Run pattern search
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    progress_bar.progress(40)
                    
                    # Choose the appropriate method based on the selected analysis approach
                    analysis_method = st.session_state.analysis_method
                    if analysis_method == "Matrix Profile":
                        # Use the original Matrix Profile search
                        results = loop.run_until_complete(
                            find_similar_patterns(
                                symbol=st.session_state.selected_coin,
                                interval=interval,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                df=df,
                                max_matches=st.session_state.max_matches,
                                following_points=pattern_length * 2,
                                search_start=st.session_state.search_range['start_date'],
                                search_end=st.session_state.search_range['end_date']
                            )
                        )
                    else:
                        # Use the feature extraction approach with PCA
                        results = loop.run_until_complete(
                            find_similar_patterns_feature_based(
                                symbol=st.session_state.selected_coin,
                                interval=interval,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                df=df,
                                max_matches=st.session_state.max_matches,
                                following_points=pattern_length * 2,
                                search_start=st.session_state.search_range['start_date'],
                                search_end=st.session_state.search_range['end_date'],
                                n_components=st.session_state.n_components
                            )
                        )
                
                progress_bar.progress(80)
                
                # Store results
                st.session_state.search_results = results
                
                # Check for error in results
                if 'error' in results:
                    error_msg = results['error']
                    progress_bar.progress(100, text=f"Error: {error_msg}")
                    time.sleep(1.5)
                    st.error(error_msg)
                # Display debug info if available
                elif 'debug_info' in results:
                    debug = results['debug_info']
                    st.session_state.debug_info = debug
                    progress_bar.progress(90)
                    
                    # Different text based on the method used
                    if 'stumpy_matches_found' in debug:
                        progress_text = f"Found {debug['stumpy_matches_found']} raw matches, kept {debug['unique_matches']} after filtering, showing {debug['final_matches']} matches"
                    elif 'feature_matches_found' in debug:
                        progress_text = f"Found {debug['feature_matches_found']} raw matches, kept {debug['unique_matches']} after filtering, showing {debug['final_matches']} matches"
                    else:
                        progress_text = f"Found matches! Kept {debug['unique_matches']} after filtering, showing {debug['final_matches']} matches"
                    
                    progress_bar.progress(100, text=progress_text)
                    time.sleep(1.5)
                else:
                    progress_bar.progress(100)
                    time.sleep(0.5)
                
                progress_bar.empty()
                
                # Force a rerun to refresh the UI with results
                st.rerun()

# Display results if available
if st.session_state.search_results:
    results = st.session_state.search_results
    
    if "error" in results:
        st.error(results["error"])
    else:
        # Convert to pandas DataFrame for the source pattern
        df = st.session_state.btc_data
        start_idx = st.session_state.selected_range['start_idx']
        end_idx = st.session_state.selected_range['end_idx']
        
        # Get the source pattern data
        source_pattern = df.iloc[start_idx:end_idx+1][["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        
        # Display feature space visualization if using feature extraction
        if results['type'] == 'feature_pattern' and 'vis_data' in results and 'match_indices' in results:
            # Get the visualization data (method is always PCA now)
            vis_data = results['vis_data']
            match_indices = results['match_indices']
            
            # Create feature space visualization
            try:
                from visualization import create_feature_space_visualization, create_feature_importance_chart
                
                # Feature space plot with PCA
                feature_space_fig = create_feature_space_visualization(
                    vis_data=vis_data,
                    feature_method="PCA",
                    match_indices=match_indices,
                    dark_mode=True
                )
                
                # Create feature importance chart if available - commented out but kept for future use
                feature_importance_fig = None
                # if 'feature_importance' in vis_data and vis_data['feature_importance'] is not None:
                #     feature_importance_fig = create_feature_importance_chart(
                #         feature_importance=vis_data['feature_importance'],
                #         top_n=10,
                #         dark_mode=True
                #     )
                
                # Display the feature space visualization using full width
                st.subheader("Pattern Families in Feature Space")
                st.plotly_chart(feature_space_fig, use_container_width=True)
                
                # Code for displaying feature importance in second column - commented out but kept for future use
                # if feature_importance_fig is not None:
                #     with st.container():
                #         st.subheader("Top Features")
                #         st.plotly_chart(feature_importance_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create feature visualization: {str(e)}")
        
        # All matches returned from backend already exclude exact matches (score <= 0.01)
        # We just need to use them directly
        temp_matches = results['flashback_patterns']
        
        # Handle filtering differently based on the analysis method
        if results['type'] == 'feature_pattern':
            # For feature extraction, don't apply multiplier filtering - use all matches
            filtered_matches = temp_matches
        else:
            # For Matrix Profile, apply the usual score multiplier filter
            if temp_matches:
                # Get the best (lowest) distance score
                best_score = min(match['distance'] for match in temp_matches)
                
                # Calculate the maximum allowed score based on percentage
                max_allowed_score = best_score * (st.session_state.max_distance_pct / 100.0)
                
                # Filter out matches with scores above the maximum allowed
                filtered_matches = [match for match in temp_matches 
                                if match['distance'] <= max_allowed_score]
            else:
                filtered_matches = []
        
        # Display the total count of matches
        total_matches = len(filtered_matches)
        
        # Calculate post-pattern price direction statistics
        higher_count = 0
        lower_count = 0
        
        if total_matches > 0:
            # Get the pattern length (to know where the blue line is)
            pattern_length = len(source_pattern)
            
            # Loop through each match to analyze what happened after
            for match in filtered_matches:
                match_data = pd.DataFrame(match["pattern_data"])
                
                # Only analyze if we have enough data after the pattern
                if len(match_data) > pattern_length:
                    # Calculate median close price of the pattern part (before the blue line)
                    pattern_close_prices = match_data.iloc[:pattern_length]['close']
                    pattern_median_close = pattern_close_prices.median()
                    
                    # Calculate median close price of the future part (after the blue line)
                    future_close_prices = match_data.iloc[pattern_length:]['close']
                    future_median_close = future_close_prices.median()
                    
                    # Compare median values to see if price level went up or down after pattern
                    if future_median_close > pattern_median_close:
                        higher_count += 1
                    elif future_median_close < pattern_median_close:
                        lower_count += 1
                    # Equal median prices not counted in either category
            
            # Display statistics with the total count of matches and filtering info
            if temp_matches and filtered_matches:
                best_score = min(match['distance'] for match in temp_matches)
                max_allowed_score = best_score * (st.session_state.max_distance_pct / 100.0)
                initial_matches = len(temp_matches)
                
                # Calculate multiplier for display
                multiplier = st.session_state.max_distance_pct / 100.0
                
                # Display pattern count header
                st.header(f"Found {total_matches} Similar Patterns")
                
                # Create a 40/60 layout with info on left and distribution chart on right
                left_right_cols = st.columns([2, 3])
                
                # LEFT SIDE - Results info and metrics
                with left_right_cols[0]:
                    # Display debug info if available
                    if 'debug_info' in st.session_state and 'search_results' in st.session_state:
                        debug = st.session_state.debug_info
                        requested = debug['requested_matches']
                        unique_matches = debug.get('unique_matches', initial_matches)
                        
                        # Add detailed caption with filtering information
                        limited_out = unique_matches - total_matches
                        # Get the highest score in the filtered matches
                        if filtered_matches:
                            highest_score = max(match['distance'] for match in filtered_matches)
                        else:
                            highest_score = 0
                        
                        if results['type'] == 'feature_pattern':
                            # For feature extraction, show variance if available
                            explained_variance = results.get('vis_data', {}).get('explained_variance')
                            components = results.get('n_components', 2)
                            
                            if explained_variance is not None:
                                if limited_out > 0:
                                    st.markdown(f"<span style='color: #cccccc;'>Feature distance score range: {best_score:.4f} to {highest_score:.4f}. Explained variance: {explained_variance:.1%} with {components} components. {unique_matches} unique patterns found, showing top {total_matches}.</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='color: #cccccc;'>Feature distance score range: {best_score:.4f} to {highest_score:.4f}. Explained variance: {explained_variance:.1%} with {components} components. {unique_matches} unique patterns found.</span>", unsafe_allow_html=True)
                            else:
                                if limited_out > 0:
                                    st.markdown(f"<span style='color: #cccccc;'>Feature distance score range: {best_score:.4f} to {highest_score:.4f}. {unique_matches} unique patterns found, showing top {total_matches}.</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='color: #cccccc;'>Feature distance score range: {best_score:.4f} to {highest_score:.4f}. {unique_matches} unique patterns found.</span>", unsafe_allow_html=True)
                        else:
                            # For Matrix Profile, show the multiplier
                            if limited_out > 0:
                                st.markdown(f"<span style='color: #cccccc;'>Shape distance score range: {best_score:.4f} to {max_allowed_score:.4f} ({multiplier:.1f}x multiplier). {unique_matches} unique patterns found, showing top {total_matches}, highest score {highest_score:.4f}.</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color: #cccccc;'>Shape distance score range: {best_score:.4f} to {max_allowed_score:.4f} ({multiplier:.1f}x multiplier). {unique_matches} unique patterns found, highest score {highest_score:.4f}.</span>", unsafe_allow_html=True)
                    else:
                        # Get the highest score in the filtered matches
                        if filtered_matches:
                            highest_score = max(match['distance'] for match in filtered_matches)
                            
                            if results['type'] == 'feature_pattern':
                                # For feature extraction, show variance if available
                                explained_variance = results.get('vis_data', {}).get('explained_variance')
                                components = results.get('n_components', 2)
                                
                                if explained_variance is not None:
                                    st.markdown(f"<span style='color: #cccccc;'>Feature distance score range: {best_score:.4f} to {highest_score:.4f}. Explained variance: {explained_variance:.1%} with {components} components.</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='color: #cccccc;'>Feature distance score range: {best_score:.4f} to {highest_score:.4f}.</span>", unsafe_allow_html=True)
                            else:
                                # For Matrix Profile, show the multiplier
                                st.markdown(f"<span style='color: #cccccc;'>Shape distance score range: {best_score:.4f} to {max_allowed_score:.4f} ({multiplier:.1f}x multiplier), highest score {highest_score:.4f}</span>", unsafe_allow_html=True)
                        else:
                            if results['type'] == 'feature_pattern':
                                explained_variance = results.get('vis_data', {}).get('explained_variance')
                                components = results.get('n_components', 2)
                                
                                if explained_variance is not None:
                                    st.markdown(f"<span style='color: #cccccc;'>Feature distance score range: {best_score:.4f}. Explained variance: {explained_variance:.1%} with {components} components.</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='color: #cccccc;'>Feature distance score range: {best_score:.4f}.</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color: #cccccc;'>Shape distance score range: {best_score:.4f} to {max_allowed_score:.4f} ({multiplier:.1f}x multiplier)</span>", unsafe_allow_html=True)
                    
                    # Add small spacer
                    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
                    
                    # Add price direction statistics
                    # Show price direction statistics if we have any valid comparisons
                    if higher_count + lower_count > 0:
                        higher_pct = (higher_count / (higher_count + lower_count)) * 100
                        lower_pct = (lower_count / (higher_count + lower_count)) * 100
                        
                        # Calculate the time duration of the "what comes next" section
                        # This is based on the interval and number of future candles (2x pattern length)
                        pattern_candles = len(source_pattern)
                        future_candles = pattern_candles * 2  # Because we're using 1/3 pattern, 2/3 future
                        
                        # Map interval to time units
                        interval_time_map = {
                            "1m": {"unit": "minutes", "value": 1},
                            "3m": {"unit": "minutes", "value": 3},
                            "5m": {"unit": "minutes", "value": 5},
                            "15m": {"unit": "minutes", "value": 15},
                            "30m": {"unit": "minutes", "value": 30},
                            "1h": {"unit": "hours", "value": 1},
                            "4h": {"unit": "hours", "value": 4},
                            "1d": {"unit": "days", "value": 1},
                            "1w": {"unit": "weeks", "value": 1}
                        }
                        
                        # Get the time value and unit for the current interval
                        time_info = interval_time_map.get(interval, {"unit": "candles", "value": 1})
                        time_unit = time_info["unit"]
                        time_value = time_info["value"]
                        
                        # Calculate total future time in the appropriate unit
                        total_future_time = future_candles * time_value
                        
                        # Format the time label with appropriate units
                        if time_unit == "minutes" and total_future_time >= 60:
                            # Convert to hours if minutes exceed 60
                            hours = total_future_time // 60
                            minutes = total_future_time % 60
                            if minutes > 0:
                                time_label = f"{hours} hours {minutes} minutes"
                            else:
                                time_label = f"{hours} hours"
                        elif time_unit == "hours" and total_future_time >= 24:
                            # Convert to days if hours exceed 24
                            days = total_future_time // 24
                            hours = total_future_time % 24
                            if hours > 0:
                                time_label = f"{days} days {hours} hours"
                            else:
                                time_label = f"{days} days"
                        else:
                            # Use the calculated time directly
                            time_label = f"{total_future_time} {time_unit}"
                        
                        # Use colored metrics with time information
                        st.metric(
                            label=f"Median Price Higher {time_label} After Pattern", 
                            value=f"{higher_count} patterns", 
                            delta=f"{higher_pct:.1f}%",
                            delta_color="normal"
                        )
                        
                        st.metric(
                            label=f"Median Price Lower {time_label} After Pattern", 
                            value=f"{lower_count} patterns", 
                            delta=f"{lower_pct:.1f}%",
                            delta_color="inverse"  # Inverse makes down neutral/up green
                        )
                
                # RIGHT SIDE - Distribution chart
                with left_right_cols[1]:
                    # Create a histogram of scores to show distribution
                    if initial_matches > 1:  # Only show if we have multiple matches
                        # Get the filtered (unique, after time-proximity) matches only
                        if 'filtered_matches_scores' in results:
                            # Use the scores after time-proximity filtering (unique matches)
                            unique_scores = results['filtered_matches_scores']
                        else:
                            # Fallback to temp_matches
                            unique_scores = [match['distance'] for match in temp_matches]
                    
                        # Find which scores were included in final result (after multiplier and limit)
                        included_scores = [match['distance'] for match in filtered_matches]
                        
                        # Calculate excluded scores (only those excluded by multiplier or limit, not time-proximity)
                        excluded_scores = [s for s in unique_scores if s not in included_scores]
                        
                        # Determine the score range for better x-axis limits
                        if unique_scores:
                            min_score = min(unique_scores) * 0.9  # Add 10% padding
                            max_score = max(unique_scores) * 1.1  # Add 10% padding
                        else:
                            min_score = 0
                            max_score = 10
                        
                        # Create the histogram with proper sizing
                        hist_fig = go.Figure()
                        
                        # Find min and max scores to set proper x-axis range
                        if unique_scores:
                            min_score = min(unique_scores)
                            max_score = max(unique_scores)
                            
                            # Add 5% margin to both sides to ensure no cut-off
                            margin = (max_score - min_score) * 0.05
                            plot_min = min_score - margin
                            plot_max = max_score + margin
                        else:
                            plot_min = 0
                            plot_max = 1
                        
                        # Define common histogram parameters to ensure consistency
                        histogram_params = {
                            'nbinsx': 30,             # Same number of bins for both
                            'opacity': 0.7,           # Same opacity for both
                            'showlegend': False,      # No legend for either
                            'autobinx': False,        # Disable auto-binning to ensure consistency
                            'xbins': {                # Explicitly define bins to be identical
                                'start': plot_min,
                                'end': plot_max,
                                'size': (plot_max - plot_min) / 30  # 30 equal bins across range
                            }
                        }
                        
                        # Add trace for included scores (blue)
                        hist_fig.add_trace(go.Histogram(
                            x=included_scores,
                            name="Included",
                            marker_color='rgba(66, 135, 245, 0.7)',
                            **histogram_params
                        ))
                        
                        # Add trace for excluded scores (black/gray)
                        if excluded_scores:
                            hist_fig.add_trace(go.Histogram(
                                x=excluded_scores,
                                name="Excluded",
                                marker_color='rgba(100, 100, 100, 0.7)',
                                **histogram_params
                            ))
                        
                        # Print debug info
                        st.session_state.excluded_count = len(excluded_scores)
                        st.session_state.included_count = len(included_scores)
                        
                        # Removed the red vertical line for a cleaner look
                        
                        # Add annotation to show counts - now only showing unique patterns after time-filtering
                        total_patterns = len(included_scores) + len(excluded_scores)
                        hist_fig.add_annotation(
                            x=0.5, 
                            y=1.05,
                            xref="paper",
                            yref="paper",
                            text=f"Blue: {len(included_scores)} included / Gray: {len(excluded_scores)} excluded",
                            showarrow=False,
                            font=dict(size=10, color="#999999")  # Match the standard text color
                        )
                        
                        # Layout for the histogram with improved size and ranges
                        score_type = "Feature Distance" if results['type'] == 'feature_pattern' else "Shape Distance"  
                        hist_fig.update_layout(
                            title=f"{score_type} Distribution",
                            title_font=dict(color="#999999", size=14),  # Match title with grid lines
                            xaxis_title="Score",
                            yaxis_title="Count",
                            height=300,  # Taller for better visibility
                            margin=dict(l=5, r=5, t=10, b=5),  # Reduced margins by 50%
                            barmode='overlay',
                            plot_bgcolor=st.session_state.candle_style['background_color'],
                            paper_bgcolor=st.session_state.candle_style['background_color'],
                            font=dict(color="#999999", size=10),  # Lighter color for axis text
                            showlegend=False,  # Remove legend entirely
                            # Enhance x-axis with proper range limits and more granular ticks
                            xaxis=dict(
                                range=[plot_min, plot_max],  # Set range to min/max with margins
                                showgrid=True,
                                gridcolor='#999999',  # Match grid color to text color
                                gridwidth=0.5,      # Make grid lines slightly thicker
                                zeroline=False,
                                dtick=0.1,  # Moderate tick intervals (0.1 steps)
                                tickformat=".2f",  # Show two decimal places for more precision
                                title_font=dict(color="#999999"),  # Lighter color for axis title
                                title="Score",      # Add explicit axis title
                                nticks=15           # Fewer tick marks for better readability
                            ),
                            # Enhance y-axis with fixed range to prevent empty space
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='#999999',  # Match grid color to text color
                                gridwidth=0.5,      # Make grid lines slightly thicker 
                                zeroline=False,
                                title_font=dict(color="#999999"),  # Lighter color for axis title
                                title="Count",      # Add explicit axis title
                                automargin=True     # Ensure labels fit
                            )
                        )
                    
                        # Display the histogram
                        st.plotly_chart(hist_fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.header(f"Found {total_matches} Similar Patterns")
            
            # Price direction stats now displayed earlier in the UI flow
            
            # First display source pattern for reference - with some space
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            source_fig = go.Figure()
            source_times = [datetime.fromtimestamp(ts/1000) for ts in source_pattern["timestamp"]]
            
            # Add source pattern on the left half with blue theme to match result charts
            source_fig.add_trace(
                go.Candlestick(
                    x=source_times,
                        open=source_pattern['open'],
                        high=source_pattern['high'],
                        low=source_pattern['low'],
                        close=source_pattern['close'],
                        name=None,
                        increasing=dict(
                            line=dict(color='rgba(66, 135, 245, 0.7)'),  # Light blue
                            fillcolor='rgba(66, 135, 245, 0.5)'          # Light blue with transparency
                        ),
                        decreasing=dict(
                            line=dict(color='rgba(26, 86, 196, 0.7)'),   # Darker blue
                            fillcolor='rgba(26, 86, 196, 0.5)'           # Darker blue with transparency
                        ),
                        hoverinfo="x+y"
                    )
            )
            
            # For source pattern, we need to ensure yellow line is at 50% and equal space on both sides
            pattern_candles = len(source_times)
            
            # Simpler approach - directly use the pattern timespan
            pattern_timespan = source_times[-1] - source_times[0]
            
            # For the source pattern, we now want 2x timespan after the yellow line
            # This ensures the yellow line is at 1/3 with more space for future data
            future_end = source_times[-1] + (pattern_timespan * 2)
            
            # Add invisible points at strategic locations to force chart expansion
            # Create invisible points to properly size the chart
            source_fig.add_trace(
                go.Scatter(
                    x=[
                        source_times[-1] + timedelta(hours=1),  # Just after pattern ends
                        future_end,                             # At the calculated future end
                        future_end + timedelta(seconds=1)       # Slightly beyond to force expansion
                    ],
                    y=[source_pattern['close'].iloc[-1]] * 3,  # Same y-value for all points
                    mode='markers',
                    marker=dict(opacity=0),  # Invisible marker
                    showlegend=False,
                    hoverinfo='none'
                )
            )
            
            # Add vertical line at the end of source pattern to match the result charts
            # This is where "what comes next" would be shown if we had future data
            future_start = source_times[-1]  # Last candle in the source pattern
            source_fig.add_shape(
                type="line",
                x0=future_start,
                x1=future_start,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="#4287f5", width=2, dash="dash")  # Light blue instead of yellow
            )
            
            # Remove the annotation to free up chart space
            
            # Calculate y-axis limits
            source_y_min = source_pattern['low'].min() * 0.995
            source_y_max = source_pattern['high'].max() * 1.005
            
            # Set different tick spacing based on selected coin for source pattern chart
            if st.session_state.selected_coin == "ETH":
                # ETH uses smaller price increments
                tick_spacing = 50    # Grid line every 50 price level for ETH
                tickformatstops_config = [
                    dict(dtickrange=[None, 250], value=",.0f"),   # Show every 50 tick label
                    dict(dtickrange=[250, None], value=",.0f")    # Show only every 250 tick label
                ]
            else:
                # BTC uses larger price increments
                tick_spacing = 1000  # Grid line every 1k price level for BTC
                tickformatstops_config = [
                    dict(dtickrange=[None, 5000], value=",.0f"),  # Show every 1k tick label
                    dict(dtickrange=[5000, None], value=",.0f")   # Show only every 5k tick label
                ]
            
            # Ultra-minimal chart layout with forced full width utilization
            source_fig.update_layout(
                height=400,  # Match with result charts
                yaxis=dict(
                    range=[source_y_min, source_y_max],
                    showgrid=True,  # Add horizontal grid lines
                    gridcolor=st.session_state.candle_style['grid_color'],
                    gridwidth=0.5,
                    title=None,      # No title
                    fixedrange=True,  # Prevent y-axis zooming which can affect layout
                    dtick=tick_spacing,  # Dynamic grid line spacing based on coin
                    tickmode="linear",  # Use linear tick mode
                    tick0=0,  # Start from 0
                    tickformat=",.0f",  # No decimal places, comma for thousands
                    tickformatstops=tickformatstops_config,  # Dynamic tick format stops based on coin
                    tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),  # Consistent grey color
                    tickcolor="#999999"  # Ensure tick marks are grey
                ),
                hoverlabel=dict(
                    bgcolor=st.session_state.candle_style['background_color'],
                    font_size=14, 
                    font_family="ProtoMono-Light, monospace"
                ),
                hovermode='x unified',
                xaxis=dict(
                    # Add 5% padding to ensure candles aren't cut off
                    range=[source_times[0] - (pattern_timespan * 0.05), future_end + (pattern_timespan * 0.05)],  # Add padding on both sides
                    title=None,      # No title
                    fixedrange=True, # Prevent x-axis zooming which can affect layout
                    automargin=False, # Don't add margins automatically
                    domain=[0, 1],    # Force the axis to use full width
                    showgrid=True,   # Add vertical grid lines
                    gridcolor=st.session_state.candle_style['grid_color'],
                    gridwidth=0.5,
                    tickangle=0,    # Flat time labels
                    tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),  # Consistent grey color
                    tickformat="%m-%d<br>%H:%M", # Two-line format
                    nticks=15,  # Increase number of x-axis divisions
                    ticks="outside",  # Place ticks outside the chart
                    ticklen=8,  # Longer tick marks
                    minor_showgrid=True,  # Show minor grid lines too
                    minor_gridcolor=st.session_state.candle_style['grid_color'],  # Ensure minor gridlines match color
                    tickcolor="#999999"  # Ensure tick marks are grey
                ),
                title=None,        # No title
                showlegend=False,  # No legend
                xaxis_rangeslider_visible=False,
                plot_bgcolor=st.session_state.candle_style['background_color'],
                paper_bgcolor=st.session_state.candle_style['background_color'],
                font=dict(family="ProtoMono-Light, monospace", color='white'),
                margin=dict(l=5, r=5, t=0, b=50),  # Reduced left/right margins by 50%
                autosize=True      # Ensure the plot uses all available space
            )
            
            # Display source pattern with title and force full container width
            st.write("**Source Pattern (Reference)**")
            # Pass explicit width of 100% to ensure chart uses all available space
            st.plotly_chart(source_fig, use_container_width=True, config={'displayModeBar': False})
            
            # Add a small spacer before the charts
            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            
            # Precompute the median price comparison for each match
            match_median_results = {}
            for match_idx, match in enumerate(filtered_matches):
                match_data = pd.DataFrame(match["pattern_data"])
                
                # Only analyze if we have enough data after the pattern
                if len(match_data) > pattern_length:
                    # Calculate median close price of the pattern part (before the blue line)
                    pattern_close_prices = match_data.iloc[:pattern_length]['close']
                    pattern_median_close = pattern_close_prices.median()
                    
                    # Calculate median close price of the future part (after the blue line)
                    future_close_prices = match_data.iloc[pattern_length:]['close']
                    future_median_close = future_close_prices.median()
                    
                    # Store comparison result for this match
                    if future_median_close > pattern_median_close:
                        match_median_results[match_idx] = "Higher"
                    elif future_median_close < pattern_median_close:
                        match_median_results[match_idx] = "Lower"
                    else:
                        match_median_results[match_idx] = "Unchanged"
                else:
                    match_median_results[match_idx] = "N/A"  # Not enough future data
            
            # Show all matches directly
            for i, match in enumerate(filtered_matches):
                # Get the median comparison result for this match
                median_result = match_median_results.get(i, "N/A")
                
                # Style the result with color
                if median_result == "Higher":
                    median_result_styled = f"<span style='color:#4CAF50'>Higher</span>"  # Green
                elif median_result == "Lower":
                    median_result_styled = f"<span style='color:#F44336'>Lower</span>"   # Red
                else:
                    median_result_styled = f"<span style='color:#999999'>{median_result}</span>"  # Grey
                
                # Header with match info and median comparison - with different score label based on analysis method
                score_label = "Feature Distance" if results['type'] == 'feature_pattern' else "Shape Distance"
                st.markdown(
                    f"**Match #{i+1}**: {datetime.fromisoformat(match['start_time'].replace('Z', '')).strftime('%Y-%m-%d')} "
                    f"({score_label}: {match['distance']:.4f}). Past/future median: {median_result_styled}", 
                    unsafe_allow_html=True
                )
                
                # Create individual figure for this match
                match_fig = go.Figure()
                
                # Get match data
                match_data = pd.DataFrame(match["pattern_data"])
                match_times = [datetime.fromtimestamp(ts/1000) for ts in match_data["timestamp"]]
                pattern_length = len(source_pattern)
                
                # Split the match data into pattern part and future part
                pattern_data = match_data.iloc[:pattern_length] if len(match_data) > pattern_length else match_data
                
                # For future data, take everything after pattern_length-1 (with overlap on yellow line)
                # but create a separate variable for display that might include extrapolated points
                future_data = match_data.iloc[pattern_length-1:] if len(match_data) > pattern_length else pd.DataFrame()
                
                pattern_times = match_times[:pattern_length] if len(match_times) > pattern_length else match_times
                future_times = match_times[pattern_length-1:] if len(match_times) > pattern_length else []
                
                # Add candlestick chart for the pattern part with blue theme
                match_fig.add_trace(
                    go.Candlestick(
                        x=pattern_times,
                        open=pattern_data['open'],
                        high=pattern_data['high'],
                        low=pattern_data['low'],
                        close=pattern_data['close'],
                        name=None,
                        increasing=dict(
                            line=dict(color='rgba(66, 135, 245, 0.7)'),  # Light blue
                            fillcolor='rgba(66, 135, 245, 0.5)'          # Light blue with transparency
                        ),
                        decreasing=dict(
                            line=dict(color='rgba(26, 86, 196, 0.7)'),   # Darker blue
                            fillcolor='rgba(26, 86, 196, 0.5)'           # Darker blue with transparency
                        )
                    )
                )
                
                # Add the future part with normal colors if available
                if not future_data.empty:
                    match_fig.add_trace(
                        go.Candlestick(
                            x=future_times,
                            open=future_data['open'],
                            high=future_data['high'],
                            low=future_data['low'],
                            close=future_data['close'],
                            name=None,
                            increasing=dict(
                                line=dict(color=st.session_state.candle_style['increasing_line_color']),
                                fillcolor=st.session_state.candle_style['increasing_color']
                            ),
                            decreasing=dict(
                                line=dict(color=st.session_state.candle_style['decreasing_line_color']),
                                fillcolor=st.session_state.candle_style['decreasing_color']
                            )
                        )
                    )
                
                # Add vertical line to separate match from future - place at the pattern end
                # Define default view range (will be adjusted with padding later)
                view_end = None  # Will be defined below
                
                if len(match_times) > pattern_length:
                    # Find the split point exactly at the end of pattern
                    future_start = match_times[pattern_length-1]
                    
                    # Calculate pattern duration for reference
                    pattern_duration = match_times[pattern_length-1] - match_times[0]
                    
                    # Calculate exactly how many candles we have after the yellow line
                    future_candles = len(match_times) - pattern_length
                    
                    # For the [xxxx____] issue, we need to make sure data VISUALLY extends to fill the space
                    # We'll add extra "invisible" data points to ensure the chart is forced to extend
                    
                    # Calculate time spans
                    pattern_timespan = future_start - match_times[0]
                    
                    # Calculate ideal size of future section - it should be exactly the same as pattern section
                    # Check how many future candles we actually have
                    future_candles_expected = pattern_length
                    future_candles_actual = len(match_times) - pattern_length
                    
                    # Force the future section to take the same visual space as the pattern section
                    # Even if we have fewer candles, we'll extend the chart to maintain visual balance
                    future_end_time = None
                    
                    if future_candles_actual >= 1:
                        if future_candles_actual >= pattern_length:
                            # We now want 2x more future candles than pattern candles
                            # Calculate where the 2x point should be
                            if future_candles_actual >= pattern_length * 2:
                                # We have enough candles for 2x (ideal case)
                                future_end_time = match_times[pattern_length + (pattern_length * 2) - 1]
                            else:
                                # We have enough for pattern_length but not full 2x, show what we have
                                future_end_time = match_times[-1]
                        else:
                            # We have some future candles but fewer than expected
                            # Estimate where the last candle should be based on timespan
                            # We now want 2x future timespan
                            avg_candle_time = pattern_timespan / (pattern_length - 1) if pattern_length > 1 else timedelta(hours=1)
                            future_end_time = future_start + (avg_candle_time * (pattern_length - 1) * 2)
                    else:
                        # No future candles, estimate based on 2x pattern timespan
                        future_end_time = future_start + (pattern_timespan * 2)
                    
                    # Add 5% padding on each side to ensure first and last candles are fully visible
                    padding_time = pattern_timespan * 0.05
                    view_start = match_times[0] - padding_time  # Add padding to start
                    view_end = future_end_time + padding_time  # Add padding to end
                    
                    # Create an invisible point at the extreme right to force the chart to expand
                    # This is a hack to make Plotly render the full width
                    invisible_future_point = view_end + timedelta(seconds=1)
                    
                    # Add an invisible point at the far right to force the chart to expand to full width
                    match_fig.add_trace(
                        go.Scatter(
                            x=[invisible_future_point],
                            y=[match_data['close'].iloc[-1]],  # Use last close price
                            mode='markers',
                            marker=dict(opacity=0),  # Completely invisible
                            showlegend=False,
                            hoverinfo='none'
                        )
                    )
                    
                    # Add blue line at the pattern end point
                    match_fig.add_shape(
                        type="line",
                        x0=future_start,
                        x1=future_start,
                        y0=0,
                        y1=1,
                        yref="paper",
                        line=dict(color="#4287f5", width=2, dash="dash")  # Light blue instead of yellow
                    )
                    
                    # Remove the annotation to free up chart space
                    
                    # To force exact 50/50 split with the yellow line in the middle without any empty space
                    # We need to set the range with very specific constraints
                    
                    # Force the exact x-axis range without allowing Plotly to adjust it
                    # This is critical - we need to set x-axis range together with other layout settings
                    # rather than as a separate call
                
                # Auto-adjust y-axis for better visibility - include ALL data
                # Use min/max for both pattern and future to ensure proper y-axis scaling
                y_min = match_data['low'].min() * 0.995
                y_max = match_data['high'].max() * 1.005
                
                # Set different tick spacing based on selected coin for individual match charts
                if st.session_state.selected_coin == "ETH":
                    # ETH uses smaller price increments
                    tick_spacing = 50    # Grid line every 50 price level for ETH
                    tickformatstops_config = [
                        dict(dtickrange=[None, 250], value=",.0f"),   # Show every 50 tick label
                        dict(dtickrange=[250, None], value=",.0f")    # Show only every 250 tick label
                    ]
                else:
                    # BTC uses larger price increments
                    tick_spacing = 1000  # Grid line every 1k price level for BTC
                    tickformatstops_config = [
                        dict(dtickrange=[None, 5000], value=",.0f"),  # Show every 1k tick label
                        dict(dtickrange=[5000, None], value=",.0f")   # Show only every 5k tick label
                    ]
                
                # Ultra-minimal chart layout with forced full width utilization
                match_fig.update_layout(
                    height=400,  # Normal height
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(
                        range=[y_min, y_max],
                        showgrid=True,  # Add horizontal grid lines
                        gridcolor=st.session_state.candle_style['grid_color'],
                        gridwidth=0.5,
                        title=None,     # No title
                        fixedrange=True,  # Prevent y-axis zooming which can affect layout
                        dtick=tick_spacing,  # Dynamic grid line spacing based on coin
                        tickmode="linear",  # Use linear tick mode
                        tick0=0,  # Start from 0
                        tickformat=",.0f",  # No decimal places, comma for thousands
                        tickformatstops=tickformatstops_config,  # Dynamic tick format stops based on coin
                        tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),  # Consistent grey color
                        tickcolor="#999999"  # Ensure tick marks are grey
                    ),
                    hoverlabel=dict(
                        bgcolor=st.session_state.candle_style['background_color'],
                        font_size=14,
                        font_family="ProtoMono-Light, monospace"
                    ),
                    hovermode='x unified',
                    xaxis=dict(
                        # Add padding when range is defined to ensure first/last candles are fully visible
                        range=[view_start, view_end] if view_end is not None else None,  # Set range only if view_end is defined
                        title=None,     # No title
                        fixedrange=True,  # Prevent x-axis zooming which can affect layout
                        automargin=False,  # Don't add margins automatically
                        domain=[0, 1],  # Force the axis to use full width
                        showgrid=True,   # Add vertical grid lines
                        gridcolor=st.session_state.candle_style['grid_color'],
                        gridwidth=0.5,
                        tickangle=0,    # Flat time labels
                        tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),  # Consistent grey color
                        tickformat="%m-%d<br>%H:%M", # Two-line format
                        nticks=15,  # Increase number of x-axis divisions
                        ticks="outside",  # Place ticks outside the chart
                        ticklen=8,  # Longer tick marks
                        minor_showgrid=True,  # Show minor grid lines too
                        minor_gridcolor=st.session_state.candle_style['grid_color'],  # Ensure minor gridlines match color
                        tickcolor="#999999"  # Ensure tick marks are grey
                    ),
                    title=None,        # No title
                    showlegend=False,  # No legend
                    plot_bgcolor=st.session_state.candle_style['background_color'],
                    paper_bgcolor=st.session_state.candle_style['background_color'],
                    font=dict(family="ProtoMono-Light, monospace", color='white'),
                    margin=dict(l=5, r=5, t=0, b=50),  # Reduced left/right margins by 50%
                    autosize=True      # Ensure the plot uses all available space
                )
                
                # Display the match chart without controls and ensure full container width
                # Pass explicit width of 100% to ensure chart uses all available space
                st.plotly_chart(match_fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No matching patterns found.")