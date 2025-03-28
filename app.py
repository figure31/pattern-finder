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
    /* ===== VARIABLE DEFINITIONS ===== */
    :root {{
        --font-family: 'ProtoMono-Light', monospace;
        --text-color: #cccccc;
        --header-color: #dddddd;
        --bg-color: #1e1e1e;
        --button-bg: #2c2c2c;
        --button-border: #444444;
        --button-hover-bg: #3a3a3a;
        --button-hover-border: #555555;
        --grid-color: #333333;
        --label-color: #999999;
        --highlight-color: #4287f5;
        --highlight-bg: rgba(66, 135, 245, 0.2);
        --highlight-selected: rgba(66, 135, 245, 0.3);
        --dropdown-bg: #222222;
    }}

    /* ===== FONT DEFINITION ===== */
    @font-face {{
        font-family: 'ProtoMono-Light';
        src: url('data:font/otf;base64,{font_base64}') format('opentype');
        font-weight: normal;
        font-style: normal;
    }}
    
    /* ===== GLOBAL SETTINGS ===== */
    * {{
        font-family: var(--font-family) !important;
    }}
    
    body, p, div, span, label, select, input, .element-container {{
        font-size: 1rem !important;
        color: var(--text-color) !important;
    }}
    
    /* ===== LAYOUT CORE ===== */
    /* Hide unnecessary UI elements */
    #MainMenu, footer, header {{
        display: none;
    }}
    
    /* Container core structure */
    .main .block-container {{
        padding: 0.2rem !important;
        padding-top: 0 !important;
        padding-bottom: 40px !important;
        max-width: 100% !important;
        margin: 0 auto !important;
    }}
    
    .main {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}
    
    .appview-container {{
        padding-top: 0 !important;
    }}
    
    /* Background color consistency */
    body, .stPlotlyChart, .js-plotly-plot, .plot-container, .element-container, .block-container, .main {{
        background-color: var(--bg-color) !important;
    }}
    
    body {{
        padding-right: 15px !important;
    }}
    
    /* ===== TYPOGRAPHY ===== */
    /* Header styles */
    h1 {{
        font-size: 1rem !important; 
        color: var(--header-color) !important;
        margin-top: 0.2rem !important; 
        margin-bottom: 0.1rem !important;
    }}
    
    h2, h3 {{
        font-size: 1rem !important; 
        color: var(--header-color) !important;
        margin-top: 0.3rem !important; 
        margin-bottom: 0.1rem !important;
    }}
    
    /* Text elements */
    p, div.markdown-text-container {{
        margin-bottom: 8px !important;
        margin-top: 8px !important;
        padding-left: 5px !important;
    }}
    
    /* ===== STREAMLIT ELEMENT SPACING ===== */
    /* Element containers - consolidated all element-container styles */
    .element-container {{
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 5px !important;
        padding-bottom: 0 !important;
    }}
    
    .stSelectbox, .stMultiselect {{
        margin-bottom: 0.2rem !important; 
        margin-top: 0 !important;
    }}
    
    div[data-testid="stVerticalBlock"] > div {{
        padding-top: 0 !important; 
        padding-bottom: 0 !important; 
        margin-top: 0 !important; 
        margin-bottom: 0 !important;
    }}
    
    div[data-testid="stHorizontalBlock"] {{
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }}
    
    .st-emotion-cache-1wmy9hl {{
        overflow-x: hidden !important;
    }}
    
    /* ===== FORM ELEMENTS ===== */
    /* Date inputs */
    .stDateInput {{
        width: 100%;
    }}
    
    /* Text input styling to match select boxes */
    .stTextInput > div > div > input {{
        background-color: var(--button-bg) !important;
        border-color: var(--button-border) !important;
        border-width: 1px !important;
        color: var(--text-color) !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: var(--button-hover-border) !important;
        border-width: 1px !important;
        box-shadow: none !important;
    }}
    
    /* Make sure text input container has the same styling as other widgets */
    .stTextInput > div {{
        border: 1px solid var(--button-border) !important;
        border-radius: 4px !important;
    }}
    
    /* Tooltip styling */
    div[data-testid="stTooltipIcon"] span {{
        background-color: #262730;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.2rem;
    }}
    
    
    /* ===== BUTTONS & SELECTORS ===== */
    /* Button styling - consolidated */
    .stButton button {{
        width: 100%;
        color: #333333 !important;
        background-color: var(--button-bg) !important;
        border-color: var(--button-border) !important;
        height: 2.5rem !important; /* Match text input height */
        padding: 0px !important;
        line-height: 1.15 !important;
        font-size: 0.9rem !important;
    }}
    
    .stButton button:hover {{
        background-color: var(--button-hover-bg) !important;
        border-color: var(--button-hover-border) !important;
    }}
    
    /* Select box styling - consolidated */ 
    .stSelectbox [data-baseweb="select"] div, 
    .stMultiselect [data-baseweb="select"] div {{
        background-color: var(--button-bg) !important;
        border-color: var(--button-border) !important;
        color: var(--text-color) !important;
    }}
    
    .stSelectbox [data-baseweb="select"] div:hover,
    .stMultiselect [data-baseweb="select"] div:hover {{
        background-color: var(--button-hover-bg) !important;
        border-color: var(--button-hover-border) !important;
    }}
    
    /* Force select dropdown color */
    div[data-baseweb="select"] div, 
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg,
    div[data-baseweb="select"] * {{
        color: var(--text-color) !important;
        fill: var(--text-color) !important;
    }}
    
    /* Dropdown menu styling */
    div[data-baseweb="popover"] div[data-baseweb="menu"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"],
    div[data-baseweb="select-option"],
    div[data-baseweb="menu"] div,
    div[data-baseweb="menu"] li,
    div[role="option"],
    li[role="option"] {{
        background-color: var(--dropdown-bg) !important;
        border-color: var(--button-border) !important;
        color: var(--text-color) !important;
    }}
    
    /* Dropdown hover and selection states */
    div[data-baseweb="select-option"]:hover,
    div[role="option"]:hover,
    li[role="option"]:hover {{
        background-color: var(--button-hover-bg) !important;
    }}
    
    div[aria-selected="true"],
    li[aria-selected="true"] {{
        background-color: var(--highlight-selected) !important;
    }}
    
    /* Date selector sizes */
    div[data-testid="stSelectbox"] {{
        max-width: 85%;
        width: 85%;
    }}
    
    #start_year, #start_month, #start_day, #end_year, #end_month, #end_day {{
        max-width: 75%;
        width: 75%;
    }}
    
    /* Selector font sizes */
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] div,
    div[data-baseweb="select"] span, 
    div[data-baseweb="select"] div {{
        font-size: 1rem !important;
    }}
    
    /* Selection box sizes */
    div[role="listbox"] {{
        max-width: 90% !important;
    }}
    
    div.stSelectbox {{
        max-width: 90% !important;
    }}
    
    /* ===== CHART STYLING ===== */
    /* Consolidated PlotlyChart styling */
    .stPlotlyChart {{
        border: none;
        padding: 8px;
        padding-bottom: 20px !important;
        padding-right: 8px !important;
        margin-bottom: 15px !important;
        margin-top: 5px !important; /* Reduced from 10px to 5px to tighten spacing */
        margin-right: 5px !important;
        overflow: visible !important;
        box-sizing: content-box !important;
        max-width: 100% !important;
    }}
    
    /* Chart elements visibility */
    .plot-container, .plotly, .js-plotly-plot, .svg-container,
    canvas, .main-svg {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }}
    
    /* Chart label styling */
    .xtick text, .ytick text {{
        color: var(--label-color) !important;
    }}
    
    .gridlayer path {{
        stroke: var(--grid-color) !important;
    }}
    
    /* Remove unwanted subplot titles */
    .gtitle, .g-gtitle {{
        display: none !important;
    }}
    
    /* ===== MODAL STYLES ===== */
    .modal-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0,0,0,0.7);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .modal-container {{
        background-color: var(--bg-color);
        border-radius: 5px;
        width: 90%;
        max-width: 1200px;
        max-height: 90vh;
        padding: 20px;
        overflow-y: auto;
    }}
    
    .close-button {{
        float: right;
        color: white;
        cursor: pointer;
        font-size: 20px;
        margin-bottom: 15px;
    }}
    
    .match-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }}
    
    .match-title {{
        margin: 0;
        flex-grow: 1;
    }}
    
    .expand-button {{
        float: right;
        margin-left: 15px;
        color: var(--highlight-color);
        cursor: pointer;
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
    
# Initialize expanded view modal state
if 'show_modal' not in st.session_state:
    st.session_state.show_modal = False
    st.session_state.modal_match_id = None
    st.session_state.modal_match_date = None
    st.session_state.modal_match_df = None
    st.session_state.modal_distance = None

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
    
    try:
        df = await provider.get_historical_ohlcv(
            symbol=symbol, 
            interval=interval,
            start_time=start_time,
            end_time=current_time.isoformat()
        )
        
        # Check if dataframe is empty or missing expected columns
        if df is None or df.empty:
            st.error(f"Unable to fetch {symbol} data. Please try again later.")
            return pd.DataFrame()
            
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Data format error: Missing columns: {', '.join(missing_columns)}")
            st.error("API response columns: " + ", ".join(df.columns.tolist()))
            return pd.DataFrame()
        
        # Convert timestamp to datetime for better plotting
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['idx'] = range(len(df))
        
        return df
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        # Print more debug info
        st.error(f"Error details: {type(e).__name__}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Create a Plotly candlestick chart with selection capabilities
def create_candlestick_chart(df, selected_range=None, style=None):
    if df is None or df.empty:
        # Return an empty figure with a message instead of None
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available. Please try a different timeframe or check your connection.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="#999999", size=14)
        )
        empty_fig.update_layout(
            height=400,
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e"
        )
        return empty_fig, {"displayModeBar": False}
    
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
    source_idx_range=None,  # Add parameter for source index range - DEPRECATED
    use_regime_filter=False,  # Whether to filter by market regime
    regime_tolerance=0,      # How strict to be with regime matching (0=exact, 1=adjacent)
    include_neutral=True,    # Whether to include neutral regime matches
    trend_threshold=0.03,    # Threshold for price change to be considered bullish/bearish
    efficiency_threshold=0.5 # Threshold for efficiency ratio to determine trending/volatile
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
            search_end_time=search_end_iso,
            use_regime_filter=use_regime_filter,
            regime_tolerance=regime_tolerance,
            include_neutral=include_neutral,
            trend_threshold=trend_threshold,
            efficiency_threshold=efficiency_threshold
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
            search_end_time=search_end_iso,
            use_regime_filter=use_regime_filter,
            regime_tolerance=regime_tolerance,
            include_neutral=include_neutral,
            trend_threshold=trend_threshold,
            efficiency_threshold=efficiency_threshold
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
    n_components=2,         # Number of components for dimensionality reduction
    use_regime_filter=False,  # Whether to filter by market regime
    regime_tolerance=0,      # How strict to be with regime matching (0=exact, 1=adjacent)
    include_neutral=True,    # Whether to include neutral regime matches
    trend_threshold=0.03,    # Threshold for price change to be considered bullish/bearish
    efficiency_threshold=0.5 # Threshold for efficiency ratio to determine trending/volatile
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
            n_components=n_components,
            use_regime_filter=use_regime_filter,
            regime_tolerance=regime_tolerance,
            include_neutral=include_neutral,
            trend_threshold=trend_threshold,
            efficiency_threshold=efficiency_threshold
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
            n_components=n_components,
            use_regime_filter=use_regime_filter,
            regime_tolerance=regime_tolerance,
            include_neutral=include_neutral,
            trend_threshold=trend_threshold,
            efficiency_threshold=efficiency_threshold
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
                        fillcolor='rgba(66, 135, 245, 0.7)'          # Light blue with transparency
                    ),
                    decreasing=dict(
                        line=dict(color='rgba(26, 86, 196, 0.7)'),   # Darker blue
                        fillcolor='rgba(26, 86, 196, 0.7)'           # Darker blue with transparency
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
    /* ===== TITLE LAYOUT ===== */
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
            For <a href="https://x.com/standardvoids" target="_blank" style="color: #4287f5; text-decoration: none;">S&V</a>
        </h1>
        """, 
        unsafe_allow_html=True
    )

# Add ultra-minimal spacing after title and pull up controls
st.markdown(
    """
    <style>
    /* ===== LAYOUT ADJUSTMENTS ===== */
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
    /* ===== COLUMN ALIGNMENT ===== */
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
    
    /* Ensure selectbox width is consistent */
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
    
    # Check if we have a valid figure
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config=config)
    else:
        st.error("Unable to create chart. Please try different settings or refresh the page.")
    
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
        
    # Store current slider values in session state for consistent access
    end_slider_key = f"end_slider_{interval}"
    if end_slider_key not in st.session_state:
        st.session_state[end_slider_key] = default_end
    
    # Apply custom styling to make the slider more compact
    st.markdown(
        """
        <style>
        /* ===== SLIDER STYLING ===== */
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
    
    # Create unique keys for the pattern selection based on timeframe
    start_key = f"pattern_selection_start_{interval}"
    end_key = f"pattern_selection_end_{interval}"

    # Create columns for the inputs and buttons - equal width for consistent sizing
    pattern_cols = st.columns([3, 1, 1, 3, 1, 1])

    # Start index section with title
    with pattern_cols[0]:
        st.markdown("**Start Index**")

    # Empty column for alignment
    with pattern_cols[1]:
        st.write("")

    # Empty column for alignment
    with pattern_cols[2]:
        st.write("")
    
    # Create a new row for the actual inputs and buttons
    input_row = st.columns([3, 1, 1, 3, 1, 1])
    
    # Start index input field
    with input_row[0]:
        # Check if we have a stored value from button click
        if f"{start_key}_stored_value" in st.session_state:
            input_value = str(st.session_state[f"{start_key}_stored_value"])
            # Clear stored value after using it
            del st.session_state[f"{start_key}_stored_value"]
        else:
            input_value = str(default_start)
            
        start_str = st.text_input(
            "Start Index",
            value=input_value,
            label_visibility="collapsed",
            key=f"{start_key}_input",
            help="Set the start index of the pattern"
        )
        # Validate input
        try:
            start_slider = int(start_str)
            start_slider = min(max(start_slider, 0), len(df)-2)  # Ensure within valid range
        except ValueError:
            start_slider = default_start

    # Decrease start button
    with input_row[1]:
        if st.button("←", key=f"{start_key}_decrease", use_container_width=True):
            start_slider = max(start_slider - 1, 0)
            # Store the value to be used in the next render
            st.session_state[f"{start_key}_stored_value"] = start_slider
            st.rerun()

    # Increase start button
    with input_row[2]:
        if st.button("→", key=f"{start_key}_increase", use_container_width=True):
            # Use the end slider value from session state
            end_slider_key = f"end_slider_{interval}"
            end_limit = st.session_state[end_slider_key]
            start_slider = min(start_slider + 1, end_limit - 1)
            # Store the value to be used in the next render
            st.session_state[f"{start_key}_stored_value"] = start_slider
            st.rerun()

    # End index section with title
    with pattern_cols[3]:
        st.markdown("**End Index**")
    
    # Empty column for alignment
    with pattern_cols[4]:
        st.write("")

    # Empty column for alignment
    with pattern_cols[5]:
        st.write("")
    
    # Use the same input row created earlier
    # End index input field
    with input_row[3]:
        # Check if we have a stored value from button click
        if f"{end_key}_stored_value" in st.session_state:
            input_value = str(st.session_state[f"{end_key}_stored_value"])
            # Clear stored value after using it
            del st.session_state[f"{end_key}_stored_value"]
        else:
            input_value = str(default_end)
            
        end_str = st.text_input(
            "End Index",
            value=input_value,
            label_visibility="collapsed",
            key=f"{end_key}_input",
            help="Set the end index of the pattern"
        )
        # Validate input
        try:
            end_slider = int(end_str)
            end_slider = min(max(end_slider, start_slider + 1), len(df)-1)  # Ensure within valid range
        except ValueError:
            end_slider = default_end
            
        # Store the end slider value in session state for consistent access
        end_slider_key = f"end_slider_{interval}"
        st.session_state[end_slider_key] = end_slider

    # Decrease end button
    with input_row[4]:
        if st.button("←", key=f"{end_key}_decrease", use_container_width=True):
            end_slider = max(end_slider - 1, start_slider + 1)
            # Store the value to be used in the next render
            st.session_state[f"{end_key}_stored_value"] = end_slider
            # Also update the session state for consistent access
            end_slider_key = f"end_slider_{interval}"
            st.session_state[end_slider_key] = end_slider
            st.rerun()

    # Increase end button
    with input_row[5]:
        if st.button("→", key=f"{end_key}_increase", use_container_width=True):
            end_slider = min(end_slider + 1, len(df)-1)
            # Store the value to be used in the next render
            st.session_state[f"{end_key}_stored_value"] = end_slider
            # Also update the session state for consistent access
            end_slider_key = f"end_slider_{interval}"
            st.session_state[end_slider_key] = end_slider
            st.rerun()
            
    # Display range info
    st.caption(f"Valid range: 0-{len(df)-1}")
    
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
        /* ===== NUMBER INPUTS & ADDITIONAL FORM ELEMENTS ===== */
        /* Style number inputs to match buttons and select boxes */
        .stButton button,
        .stNumberInput [data-baseweb="input"],
        .stNumberInput [data-baseweb="base-input"],
        .stSelectbox [data-baseweb="select"] {
            background-color: var(--button-bg) !important;
            color: var(--text-color) !important;
            border-color: var(--button-border) !important;
            border-width: 1px !important;
            border-style: solid !important;
            border-radius: 4px !important;
        }
        
        /* Match hover states */
        .stButton button:hover,
        .stNumberInput:hover [data-baseweb="input"],
        .stNumberInput:hover [data-baseweb="base-input"],
        .stSelectbox:hover [data-baseweb="select"] {
            background-color: var(--button-hover-bg) !important;
            border-color: var(--button-hover-border) !important;
        }
        
        /* Ensure text colors match */
        .stButton button span,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] span {
            color: var(--text-color) !important;
        }
        
        /* Make labels match */
        .stNumberInput label,
        .stSelectbox label {
            color: var(--text-color) !important;
        }
        
        /* Remove spinner buttons from number inputs */
        input[type="number"]::-webkit-inner-spin-button, 
        input[type="number"]::-webkit-outer-spin-button { 
            -webkit-appearance: none !important;
            margin: 0 !important;
            opacity: 0 !important;
        }
        
        /* Number input base styling */
        .stNumberInput input {
            background-color: var(--button-bg) !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Remove artifacts */
        .stNumberInput [data-baseweb="input"]::after,
        .stNumberInput [data-baseweb="input"]::before,
        .stNumberInput [data-baseweb="base-input"]::after,
        .stNumberInput [data-baseweb="base-input"]::before {
            display: none !important;
            content: none !important;
        }
        
        /* Hide visual markers */
        [data-testid="stMarkdownContainer"] small {
            display: none !important;
        }
        
        /* ===== SLIDER FIXES ===== */
        /* Fix for temporary slider artifacts */
        div.element-container:has(div[data-baseweb="slider"]) ~ div.element-container:has(div[data-baseweb="slider"]) {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            position: absolute !important;
            pointer-events: none !important;
        }
        
        /* Hide sliders adjacent to selectboxes */
        div.element-container:has(div[data-baseweb="select"]) ~ div.element-container:has(div[data-baseweb="slider"]) {
            opacity: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
            position: absolute !important;
            z-index: -999 !important;
        }
        
        /* ===== EXPAND BUTTON FIXES ===== */
        /* Global button styling to ensure consistent button appearance */
        .stButton button {
            margin: 0 !important;
            padding: 0.55rem 0.55rem !important;
            height: 2.5rem !important;
            min-height: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Match container styling */
        .match_header_cols {
            display: flex !important;
            align-items: center !important;
        }
        
        /* Fix match header column spacing */
        .match_header_cols [data-testid="column"] {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Remove extra spacing around containers in match header */
        .match_header_cols [data-testid="stMarkdownContainer"],
        .match_header_cols [data-testid="element-container"] {
            margin-bottom: 0 !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        
        /* Ensure vertical alignment of the button */
        .match_header_cols .stButton {
            display: flex !important;
            align-items: center !important;
            height: 100% !important;
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
        # Analysis Method with help icon directly next to it
        if "analysis_method" not in st.session_state or st.session_state.analysis_method == "Matrix Profile":
            st.markdown("<div style='font-weight:bold;margin-bottom:0px;'>Analysis Method <span>&nbsp;</span></div>", unsafe_allow_html=True, help="Matrix Profile finds visually similar patterns using a shape-based matching algorithm. It's ideal for finding historical occasions where price action behaved similarly.")
        else:
            st.markdown("<div style='font-weight:bold;margin-bottom:0px;'>Analysis Method <span>&nbsp;</span></div>", unsafe_allow_html=True, help="Feature Extraction uses Principal Component Analysis (PCA) to find statistically similar patterns based on underlying market conditions.")
        
        # Function to update tooltip when method changes
        def on_method_change():
            st.session_state.needs_rerun = True
            
        analysis_method = st.selectbox(
            "Analysis Method",
            options=["Matrix Profile", "Feature Extraction"],
            index=0 if "analysis_method" not in st.session_state else 
                  (0 if st.session_state.analysis_method == "Matrix Profile" else 1),
            label_visibility="collapsed",
            on_change=on_method_change,
            key="analysis_method_select"
        )
        # Store the selected method in session state
        st.session_state.analysis_method = analysis_method
        
        # Handle rerun to update tooltip
        if st.session_state.get('needs_rerun', False):
            st.session_state.needs_rerun = False
            st.rerun()
    
    with analysis_method_cols[1]:
        # Always create a container for consistency regardless of method
        components_container = st.container()
        
        # Always create the same header structure regardless of method
        with components_container:
            # Number of Components title with help icon directly next to it
            if analysis_method == "Feature Extraction":
                st.markdown("<div style='font-weight:bold;margin-bottom:0px;'>Number of Components <span>&nbsp;</span></div>", unsafe_allow_html=True, help="Number of components for dimensionality reduction. 2-3 components visualize well, more components can capture more complex relationships.")
            else:
                # Empty div to maintain layout
                st.markdown("<div style='font-weight:bold;margin-bottom:0px;'>Number of Components</div>", unsafe_allow_html=True)
            
            # Show component selector only if Feature Extraction is selected
            if analysis_method == "Feature Extraction":
                component_options = [2, 3, 4, 5]
                n_components = st.selectbox(
                    "Components",
                    options=component_options,
                    index=component_options.index(st.session_state.n_components) if hasattr(st.session_state, 'n_components') and st.session_state.n_components in component_options else 0,
                    label_visibility="collapsed",
                    key="components_selector"  # Add explicit key to avoid conflicts
                )
                # Store the components value
                st.session_state.n_components = n_components
            else:
                # When Matrix Profile is selected, create an empty placeholder to maintain layout
                st.markdown('<div style="height:34px;"></div>', unsafe_allow_html=True)
                
    # Add CSS for checkbox alignment
    st.markdown("""
    <style>
    /* Reduce vertical space between elements */
    div.element-container {
        margin-bottom: 0.2rem !important;
    }
    
    /* Align checkbox labels in a row */
    .stCheckbox, .stRadio {
        display: inline-block !important;
        margin-right: 10px !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* Fix checkbox vertical alignment */
    .stCheckbox > div:first-child, .stRadio > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Remove extra space around markdown title */
    [data-testid="stMarkdownContainer"] {
        margin-bottom: 0 !important;
    }
    
    /* Fix radio button layout */
    .stRadio > div > div {
        flex-direction: row !important;
        gap: 10px !important;
    }
    
    /* Ensure radio button labels are aligned */
    .stRadio > div > div > label {
        display: flex !important;
        align-items: center !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add a small space before the checkbox row
    st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
    
    # Create a single row for all checkboxes side by side
    checkbox_cols = st.columns([1, 1, 1, 1])
    
    # Market Regime Filtering
    with checkbox_cols[0]:
        st.markdown("<div style='margin-bottom:0px;font-weight:bold;'>Market Regime Filtering</div>", unsafe_allow_html=True)
        use_regime_filter = st.checkbox(
            "Enable",
            value=st.session_state.get('use_regime_filter', False),
            help="Filter matches to show only those from similar market regimes as the source pattern"
        )
        st.session_state.use_regime_filter = use_regime_filter
    
    # Regime Match Strictness
    with checkbox_cols[1]:
        st.markdown("<div style='margin-bottom:0px;font-weight:bold;'>Regime Match Strictness</div>", unsafe_allow_html=True)
        regime_tolerance = st.radio(
            "Strictness",
            options=["Exact Match", "Similar Regimes"],
            index=st.session_state.get('regime_tolerance', 0),
            disabled=not use_regime_filter,
            horizontal=True,
            help="Exact match shows only patterns from the same regime. Similar regimes includes adjacent regimes.",
            label_visibility="collapsed"
        )
        # Convert UI selection to numeric value
        st.session_state.regime_tolerance = 0 if regime_tolerance == "Exact Match" else 1
    
    # Include Neutral Regime
    with checkbox_cols[2]:
        st.markdown("<div style='margin-bottom:0px;font-weight:bold;'>Include Neutral Regime</div>", unsafe_allow_html=True)
        include_neutral = st.checkbox(
            "Include",
            value=st.session_state.get('include_neutral', True),
            disabled=not use_regime_filter,
            help="Always include patterns from the neutral market regime regardless of the source pattern's regime"
        )
        st.session_state.include_neutral = include_neutral
    
    # Prediction Method
    with checkbox_cols[3]:
        st.markdown("<div style='margin-bottom:0px;font-weight:bold;'>AVG Prediction Line</div>", unsafe_allow_html=True)
        use_weighted = st.checkbox(
            "Weighted",
            value=st.session_state.get('use_weighted', False),
            help="Better matches (lower scores) have higher influence on the prediction line"
        )
        st.session_state.use_weighted = use_weighted
        
    # Add regime parameters when regime filtering is enabled
    if use_regime_filter:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        regime_param_cols = st.columns([1, 1, 2, 2])
        
        # Trend Threshold input (Bull/Bear percentage)
        with regime_param_cols[0]:
            # Trend Threshold title with help icon directly next to it
            st.markdown("<div style='font-weight:bold;margin-bottom:0px;'>Trend Threshold <span>&nbsp;</span></div>", unsafe_allow_html=True, help="Threshold for price change to be considered bullish/bearish (as decimal, e.g., 0.03 = 3%)")
            
            trend_threshold_str = st.text_input(
                "Trend Threshold",
                value=str(st.session_state.get('trend_threshold', 0.03)),
                label_visibility="collapsed",
                key="trend_threshold_input"
            )
            
            # Validate input and convert to float
            try:
                trend_threshold = float(trend_threshold_str)
                trend_threshold = min(max(trend_threshold, 0.001), 0.20)  # Clamp within range
                st.session_state.trend_threshold = trend_threshold
            except ValueError:
                trend_threshold = st.session_state.get('trend_threshold', 0.03)  # Fall back to default
                
            # Show range info
            st.caption("Range: 0.001-0.20")
            
        # Efficiency Threshold input (Trendy/Volatile threshold)
        with regime_param_cols[1]:
            # Efficiency Threshold title with help icon directly next to it
            st.markdown("<div style='font-weight:bold;margin-bottom:0px;'>Efficiency Threshold <span>&nbsp;</span></div>", unsafe_allow_html=True, help="Threshold for efficiency ratio to determine if price movement is trendy or volatile (0-1)")
            
            efficiency_threshold_str = st.text_input(
                "Efficiency Threshold",
                value=str(st.session_state.get('efficiency_threshold', 0.5)),
                label_visibility="collapsed",
                key="efficiency_threshold_input"
            )
            
            # Validate input and convert to float
            try:
                efficiency_threshold = float(efficiency_threshold_str)
                efficiency_threshold = min(max(efficiency_threshold, 0.1), 0.9)  # Clamp within range
                st.session_state.efficiency_threshold = efficiency_threshold
            except ValueError:
                efficiency_threshold = st.session_state.get('efficiency_threshold', 0.5)  # Fall back to default
                
            # Show range info
            st.caption("Range: 0.1-0.9")
        
    # Add a small space after the checkbox row
    st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
    
    # Create an info button that expands to show the text
    if analysis_method == "Matrix Profile":
        info_text = """
        **Matrix Profile - Find Visually Similar Patterns:** Matrix Profile finds visually similar patterns using a shape-based matching algorithm. It's ideal for finding historical occasions where price action behaved similarly to what you're seeing now or at a specific point in time. You can use it to confirm or refute specific chart formations you've identified. Setting tip: A multiplier filter setting of 1.2-1.5 provides strict matching for closer pattern similarity, while higher values up to 2.0 offers more flexible matching to capture broader pattern families. Score matches below 2 are ideal, above 4 the system is less reliable. For pattern selection, 15-50 candles generally works well, though up to 75 can still provide meaningful results. Beyond 100 candles, matches become increasingly rare and less precise. We recommend setting 50-100 maximum matches initially to explore similar historical periods. Important: Always examine the actual match charts rather than relying solely on the higher/lower statistics. Our median price comparison uses a fixed timeframe equal to 2 times the length of your selected pattern, which may not align with your specific trading horizon. The patterns themselves often reveal nuances and potential outcomes that statistics alone cannot capture. For trading guidance, if 70%+ of historical patterns moved in one direction, this may suggest a bias worth exploring further with your own analysis.
        """
    else:  # feature_extraction
        info_text = """
        **Feature Extraction - Find Statistically Similar Patterns:** Feature Extraction uses Principal Component Analysis (PCA) to find statistically similar patterns based on underlying market conditions. This method excels at identifying market conditions and regime changes through statistical relationships—analyzing volatility clustering, momentum divergences, trend strength transitions, and complex intermarket correlations. It's more statistical than visual. The scatter plot shows pattern groupings where the blue dot represents your selected pattern, teal dots show matches, and proximity indicates similarity level. Patterns are matched using statistical indicators like volatility, trend strength, and candlestick characteristics rather than visual shape. Lower distance scores (under 2.0) indicate stronger statistical similarity, with scores below 1.0 representing particularly strong matches. Unlike visual matching, these scores represent distances in high-dimensional feature space (typically 15-25 dimensions reduced through PCA)—patterns with similar volatility signatures, trend momentum, and price action characteristics cluster together regardless of visual appearance. For component selection, start with 2 components (default). If explained variance is below 65%, try increasing to 3-4 components to capture more statistical detail, though higher values may include noise. Start with 50-100 maximum matches to identify the primary pattern clusters. For deeper regime analysis or system development, increase to 200+ matches to better understand the full feature landscape. Feature extraction works best with 20-40 candle patterns that capture sufficient market behavior to extract meaningful statistical properties. Important: Always examine the actual match charts rather than relying solely on the higher/lower statistics. Our median price comparison uses a fixed timeframe equal to 2 times the length of your selected pattern, which may not align with your specific trading horizon. The patterns themselves often reveal nuances and potential outcomes that statistics alone cannot capture. While these patterns may not look the same, they often lead to similar market behavior, making this method more sophisticated than the more straightforward matrix profile analysis.
        """
    
    # Add custom CSS to fix vertical spacing between analysis method and info button
    st.markdown("""
    <style>
    /* ===== EXPANDER STYLING ===== */
    /* Adjust spacing around selectbox elements */
    div.element-container:has([data-baseweb="select"]) {
        margin-bottom: 0.2rem !important;
    }
    
    /* Target the container of the expander to match other element spacing */
    div.element-container:has(div[data-testid="stExpander"]) {
        margin-top: 0.0rem !important;
        margin-bottom: 0.2rem !important;
        padding: 0 !important;
    }
    
    /* Style the expander to match other buttons EXACTLY */
    .streamlit-expanderHeader {
        background-color: var(--button-bg) !important;
        border-color: var(--button-border) !important;
        border-radius: 4px !important;
        border-width: 1px !important;
        color: var(--text-color) !important;
        font-family: var(--font-family) !important;
        /* Match button padding exactly */
        padding: 0.375rem 0.75rem !important;
        line-height: 1 !important;
        height: 2rem !important;
        min-height: 0 !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--button-hover-bg) !important;
        border-color: var(--button-hover-border) !important;
    }
    
    /* Style the expander content area */
    .streamlit-expanderContent {
        background-color: var(--dropdown-bg) !important;
        border-color: var(--button-border) !important;
        border-width: 1px !important;
        border-top: none !important;
        padding: 10px !important;
    }
    
    /* Style the arrow icon in the expander */
    .streamlit-expanderHeader svg {
        color: var(--text-color) !important;
        fill: var(--text-color) !important;
        margin-right: 0.5rem !important;
    }
    
    /* Style the content text inside the expander */
    .streamlit-expanderContent p {
        color: var(--text-color) !important;
        margin-bottom: 5px !important;
    }
    
    /* Fix the text inside the button */
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1rem !important;
        font-weight: normal !important;
        margin: 0 !important;
        padding: 0 !important;
        color: var(--text-color) !important;
    }
    
    /* Container styling */
    div[data-testid="stExpander"] {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* Content area styling */
    div[data-testid="stExpander"] details {
        background-color: var(--bg-color) !important;
    }
    
    div[data-testid="stExpander"] details summary {
        margin-bottom: 0 !important;
        list-style: none !important;
        padding: 0 !important;
    }
    
    div[data-testid="stExpander"] details summary ~ div {
        padding: 1em !important;
        background-color: var(--bg-color) !important;
    }
    
    /* Adjust spacing around the expander */
    div[data-testid="stExpander"] .stExpander {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Only control top margin of expander container, allow bottom margin */
    div.element-container:has(div[data-testid="stExpander"]) {
        margin-top: 0 !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use the full width for the info button
    # Use Streamlit's built-in expander with simpler label and matching styling
    # Use Streamlit's built-in expander with styling that matches other buttons
    with st.expander("Info", expanded=False):
        st.markdown(info_text, unsafe_allow_html=True)
    
    # No spacer needed
    # st.markdown("<div style='height: 5px'></div>", unsafe_allow_html=True)
    
    with pattern_search_cols[0]:
        st.markdown("**Maximum Matches**")
        
        # Text input with min/max validation
        max_matches_str = st.text_input(
            "Maximum Matches",
            value=str(current_max_matches),
            label_visibility="collapsed",
            help="Set the maximum number of patterns to find (10-300). Higher values will find more patterns but may take longer to process.",
            key="max_matches_input"
        )
        
        # Validate input and convert to int
        try:
            max_matches = int(max_matches_str)
            max_matches = min(max(max_matches, 10), 300)  # Clamp within range
        except ValueError:
            max_matches = current_max_matches  # Fall back to current value on invalid input
                
        # Show range info
        st.caption("Range: 10-300")
    
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
            
            # Text input with min/max validation
            multiplier_str = st.text_input(
                "Filter Score Multiplier",
                value=str(current_multiplier),
                label_visibility="collapsed",
                help="Filter out patterns with scores above this multiplier (1.0-3.0) of the best match score. Lower values = more similar patterns only.",
                key="filter_score_input"
            )
            
            # Validate input and convert to float
            try:
                multiplier = float(multiplier_str)
                multiplier = min(max(multiplier, 1.0), 3.0)  # Clamp within range
                # Format to one decimal place for display
                multiplier = round(multiplier * 10) / 10
            except ValueError:
                multiplier = current_multiplier  # Fall back to current value on invalid input
                
            # Show range info
            st.caption("Range: 1.0-3.0x")
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
        
        # Reset all expanded views when starting a new search
        for key in list(st.session_state.keys()):
            if key.startswith('expand_state_'):
                st.session_state[key] = False
        
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
                                search_end=st.session_state.search_range['end_date'],
                                use_regime_filter=st.session_state.use_regime_filter,
                                regime_tolerance=st.session_state.regime_tolerance,
                                include_neutral=st.session_state.include_neutral,
                                trend_threshold=st.session_state.get('trend_threshold', 0.03),
                                efficiency_threshold=st.session_state.get('efficiency_threshold', 0.5)
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
                                n_components=st.session_state.n_components,
                                use_regime_filter=st.session_state.use_regime_filter,
                                regime_tolerance=st.session_state.regime_tolerance,
                                include_neutral=st.session_state.include_neutral,
                                trend_threshold=st.session_state.get('trend_threshold', 0.03),
                                efficiency_threshold=st.session_state.get('efficiency_threshold', 0.5)
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
                # Use consistent chart configuration
                chart_config = {
                    'scrollZoom': True,
                    'displaylogo': False,
                }
                st.plotly_chart(feature_space_fig, use_container_width=True, config=chart_config)
                
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
        
        # Display source pattern market regime if available
        if 'source_regime' in results and 'source_regime_name' in results:
            source_regime = results['source_regime']
            source_regime_name = results['source_regime_name']
            
            # Style regimes with colors
            regime_colors = {
                1: "#4CAF50",  # Bullish-Stable: Green
                2: "#8BC34A",  # Bullish-Volatile: Light Green
                3: "#9E9E9E",  # Neutral: Grey
                4: "#FF9800",  # Bearish-Stable: Orange
                5: "#F44336",  # Bearish-Volatile: Red
                6: "#2196F3"   # Choppy: Blue
            }
            regime_color = regime_colors.get(source_regime, "#9E9E9E")
            
            # Display the source pattern's regime with styling
            st.markdown(f"**Source Pattern Market Regime:** <span style='color:{regime_color}'>{source_regime_name}</span>", 
                        unsafe_allow_html=True)
            
            # Show regime filtering info if active
            if st.session_state.use_regime_filter:
                tolerance_text = "exact match" if st.session_state.regime_tolerance == 0 else "similar regimes"
                neutral_text = ", including neutral regime matches" if st.session_state.include_neutral else ""
                
                st.markdown(f"*Filtering active:* Showing only {tolerance_text}{neutral_text}. "
                          f"({results.get('debug_info', {}).get('regime_filtered_matches', 'N/A')} matches after filtering)")
        
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
                    # Get the last close price of the pattern part (before the blue line)
                    pattern_last_close = match_data.iloc[pattern_length-1]['close']
                    
                    # Get the last close price of the future part (after the blue line)
                    future_last_close = match_data.iloc[-1]['close']
                    
                    # Compare the last candle's close price to see if price went up or down after pattern
                    if future_last_close > pattern_last_close:
                        higher_count += 1
                    elif future_last_close < pattern_last_close:
                        lower_count += 1
                    # Equal prices not counted in either category
            
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
                            label=f"Final Price Higher {time_label} After Pattern", 
                            value=f"{higher_count} patterns", 
                            delta=f"{higher_pct:.1f}%",
                            delta_color="normal"
                        )
                        
                        st.metric(
                            label=f"Final Price Lower {time_label} After Pattern", 
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
                            # Enhance x-axis with proper range limits and adaptive ticks
                            xaxis=dict(
                                range=[plot_min, plot_max],  # Set range to min/max with margins
                                showgrid=True,
                                gridcolor='#999999',  # Match grid color to text color
                                gridwidth=0.5,      # Make grid lines slightly thicker
                                zeroline=False,
                                # Remove fixed dtick to allow auto-adjustment based on data range
                                tickformat=".2f",  # Show two decimal places for more precision
                                title_font=dict(color="#999999"),  # Lighter color for axis title
                                title="Score",      # Add explicit axis title
                                nticks=8,          # Limit number of ticks to prevent overlap
                                automargin=True    # Ensure labels don't get cut off
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
                        # Use consistent chart configuration
                        chart_config = {
                            'scrollZoom': True,
                            'displaylogo': False,
                        }
                        st.plotly_chart(hist_fig, use_container_width=True, config=chart_config)
            else:
                st.header(f"Found {total_matches} Similar Patterns")
            
            # Price direction stats now displayed earlier in the UI flow
            
            # Function to create prediction visualization chart that shows match outcomes
            def create_prediction_chart(source_pattern, matches, style=None, max_matches=30, main_df=None, start_idx=None, end_idx=None, use_weighted=False):
                """
                Create a chart showing potential future outcomes based on all matches.
                
                Args:
                    source_pattern: Reference pattern data
                    matches: List of pattern matches
                    style: Candlestick style for the chart
                    max_matches: Maximum number of matches to display
                    main_df: Main dataframe for extracting actual future data
                    start_idx: Start index of pattern in main dataframe
                    end_idx: End index of pattern in main dataframe
                    use_weighted: Whether to use weighted averaging instead of median
                    matches: List of match results with pattern_data
                    style: Chart style dictionary
                    max_matches: Maximum number of matches to include in visualization
                    main_df: Optional main dataframe to extract actual future data if pattern is historical
                    start_idx: Optional start index of the pattern in the main dataframe
                    end_idx: Optional end index of the pattern in the main dataframe
                    
                Returns:
                    Plotly figure object
                """
                if style is None:
                    style = st.session_state.candle_style
                
                if not matches or len(matches) == 0:
                    return None
                
                # Create a figure for the prediction chart
                pred_fig = go.Figure()
                
                # Get the source pattern data
                source_df = pd.DataFrame(source_pattern)
                source_times = [datetime.fromtimestamp(ts/1000) for ts in source_df["timestamp"]]
                pattern_length = len(source_df)
                
                # Add source pattern as candlesticks (past data)
                pred_fig.add_trace(
                    go.Candlestick(
                        x=source_times,
                        open=source_df['open'],
                        high=source_df['high'],
                        low=source_df['low'],
                        close=source_df['close'],
                        name=None,
                        increasing=dict(
                            line=dict(color='rgba(66, 135, 245, 0.7)'),  # Light blue
                            fillcolor='rgba(66, 135, 245, 0.7)'          # Light blue with transparency
                        ),
                        decreasing=dict(
                            line=dict(color='rgba(26, 86, 196, 0.7)'),   # Darker blue
                            fillcolor='rgba(26, 86, 196, 0.7)'           # Darker blue with transparency
                        ),
                        hoverinfo="x+y"
                    )
                )
                
                # Get the reference price (last close of source pattern)
                reference_price = source_df['close'].iloc[-1]
                reference_time = source_times[-1]
                
                # Create invisible anchor point at reference price/time
                pred_fig.add_trace(
                    go.Scatter(
                        x=[reference_time],
                        y=[reference_price],
                        mode='markers',
                        marker=dict(size=10, color='rgba(255, 255, 255, 0.5)'),
                        name='Reference Point',
                        hoverinfo='y',
                        showlegend=False
                    )
                )
                
                # Calculate the average timeframe between candles in the pattern
                # This will help us space out future projections
                time_diffs = []
                for i in range(1, len(source_times)):
                    time_diffs.append((source_times[i] - source_times[i-1]).total_seconds())
                avg_time_diff = np.mean(time_diffs) if time_diffs else 3600  # Default to 1 hour if can't calculate
                
                # Calculate how far into the future we should display (2x pattern length) 
                # Define this early for consistent use throughout the function
                target_future_points = pattern_length * 2

                # Create a list to hold normalized future paths
                match_futures = []
                min_pct_changes = []  # Lowest point per match
                max_pct_changes = []  # Highest point per match
                final_pct_changes = []  # Final point per match
                
                # Limit the number of matches to show
                num_matches = min(len(matches), max_matches)
                
                # Process each match to extract future data and normalize
                for i in range(num_matches):
                    match = matches[i]
                    match_data = pd.DataFrame(match["pattern_data"])
                    
                    # Skip if not enough data
                    if len(match_data) <= pattern_length:
                        continue
                        
                    # Get the future part (everything after pattern_length)
                    future_data = match_data.iloc[pattern_length:]
                    
                    # Get the reference point (last candle of the pattern part)
                    match_reference_price = match_data['close'].iloc[pattern_length-1]
                    
                    # Calculate percentage changes from reference point
                    pct_changes = (future_data['close'] / match_reference_price - 1) * 100
                    pct_changes_list = pct_changes.tolist()
                    
                    # Store the data
                    match_futures.append({
                        'pct_changes': pct_changes_list,
                        'min_pct': pct_changes.min() if not pct_changes.empty else 0,
                        'max_pct': pct_changes.max() if not pct_changes.empty else 0,
                        'final_pct': pct_changes.iloc[-1] if not pct_changes.empty else 0,
                        'distance': match['distance']  # Keep track of match quality
                    })
                    
                    # Store extremes and final values for statistics
                    if not pct_changes.empty:
                        min_pct_changes.append(pct_changes.min())
                        max_pct_changes.append(pct_changes.max())
                        final_pct_changes.append(pct_changes.iloc[-1])
                
                # Calculate average timeframes for future projection
                projection_times = []
                # Use consistent max_future_points based on target_future_points
                max_future_points = max(
                    max([len(m['pct_changes']) for m in match_futures]) if match_futures else 0,
                    target_future_points  # Ensure at least target_future_points
                )
                
                for i in range(max_future_points):
                    projection_times.append(reference_time + timedelta(seconds=avg_time_diff * (i+1)))
                
                # Add lines for each match outcome
                for i, match_future in enumerate(match_futures):
                    pct_changes = match_future['pct_changes']
                    match_quality = match_future['distance']  # Lower distance = better match
                    
                    # Calculate match quality based on distance
                    # Get all distances for color mapping
                    all_distances = [m['distance'] for m in match_futures]
                    min_dist = min(all_distances)
                    max_dist = max(all_distances)
                    norm_range = max_dist - min_dist if max_dist > min_dist else 1
                    
                    # Normalize the match quality to a 0-1 range (0 = best, 1 = worst)
                    normalized_quality = (match_quality - min_dist) / norm_range if norm_range > 0 else 0.5
                    
                    # Create a color gradient from light to dark gray based on match quality
                    # Very light gray (235) for best matches, dark gray (50) for worst matches
                    # Use a smooth gradient between them
                    gray_level = int(235 - normalized_quality * 185)
                    
                    # Set opacity (always visible but slightly transparent)
                    opacity = 0.65
                    
                    # Convert to numpy array for easier handling
                    pct_array = np.array(pct_changes)
                    
                    # Calculate future prices based on percentage changes
                    future_prices = [reference_price * (1 + pct/100) for pct in pct_array]
                    
                    # For consistent display, use exactly the target number of data points
                    # If we have more than we need, truncate 
                    # If we have fewer than we need, we'll only draw what we have
                    data_points = min(len(pct_array), target_future_points)
                    
                    # Always use the consistent time grid for all elements (real data and projections)
                    # Get up to the target number of projection times
                    times_for_match = projection_times[:data_points]
                    
                    # Create the line color based on match quality
                    # Simple gray scale - light gray for good matches, dark for poor matches
                    line_color = f'rgba({gray_level}, {gray_level}, {gray_level}, {opacity})'
                    
                    # Add line for this match with coloring based on quality
                    pred_fig.add_trace(
                        go.Scatter(
                            x=[reference_time] + times_for_match,
                            y=[reference_price] + future_prices[:data_points],
                            mode='lines',
                            line=dict(
                                color=line_color,
                                width=1.5
                            ),
                            name=f"Match #{i+1}",
                            showlegend=False,
                            hoverinfo="y+text",
                            hovertext=["Reference"] + [f"{pct:.2f}% change" for pct in pct_array[:data_points]]
                        )
                    )
                    
                    # We'll rely on the min, max, and final points below for hover information
                    # instead of adding extra hover points here
                    
                    # Add markers for min, max, and final points
                    # Need to be careful here as we're now limiting data_points based on target_future_points
                    # Make sure we don't use indices beyond what we're actually displaying
                    displayed_pct_array = pct_array[:data_points]
                    displayed_future_prices = future_prices[:data_points]
                    
                    if len(displayed_pct_array) > 0:
                        # Get indices of min and max points within the displayed part only
                        min_idx = np.argmin(displayed_pct_array)
                        max_idx = np.argmax(displayed_pct_array)
                        
                        # Min point (red)
                        # Make sure min_idx is actually valid for times_for_match
                        if min_idx < len(times_for_match):
                            pred_fig.add_trace(
                                go.Scatter(
                                    x=[times_for_match[min_idx]],
                                    y=[displayed_future_prices[min_idx]],
                                    mode='markers',
                                    marker=dict(
                                        size=6,
                                        color='rgba(239, 83, 80, {})'.format(opacity),
                                        line=dict(
                                            color='rgba(239, 83, 80, {})'.format(opacity),
                                            width=1
                                        )
                                    ),
                                    name=f"Min: {displayed_pct_array[min_idx]:.2f}%",
                                    showlegend=False,
                                    hoverinfo="text",
                                    hovertext=[f"Match #{i+1} - Min: {displayed_pct_array[min_idx]:.2f}%"]
                                )
                            )
                        
                        # Max point (green)
                        # Make sure max_idx is actually valid for times_for_match
                        if max_idx < len(times_for_match):
                            pred_fig.add_trace(
                                go.Scatter(
                                    x=[times_for_match[max_idx]],
                                    y=[displayed_future_prices[max_idx]],
                                    mode='markers',
                                    marker=dict(
                                        size=6,
                                        color='rgba(38, 166, 154, {})'.format(opacity),
                                        line=dict(
                                            color='rgba(38, 166, 154, {})'.format(opacity),
                                            width=1
                                        )
                                    ),
                                    name=f"Max: {displayed_pct_array[max_idx]:.2f}%",
                                    showlegend=False,
                                    hoverinfo="text",
                                    hovertext=[f"Match #{i+1} - Max: {displayed_pct_array[max_idx]:.2f}%"]
                                )
                            )
                        
                        # Final point (blue) - only if we have at least one point in the displayed array
                        if len(times_for_match) > 0 and len(displayed_pct_array) > 0:
                            final_pct = displayed_pct_array[-1]
                            pred_fig.add_trace(
                                go.Scatter(
                                    x=[times_for_match[-1]],
                                    y=[displayed_future_prices[-1]],
                                    mode='markers',
                                    marker=dict(
                                        size=6,
                                        color='rgba(66, 135, 245, {})'.format(opacity),
                                        line=dict(
                                            color='rgba(66, 135, 245, {})'.format(opacity),
                                            width=1
                                        )
                                    ),
                                    name=f"Final: {final_pct:.2f}%",
                                    showlegend=False,
                                    hoverinfo="text",
                                    hovertext=[f"Match #{i+1} - Final: {final_pct:.2f}%"]
                                )
                            )
                
                # Calculate the "ghost path" - either median or weighted average of all matches at each point
                if match_futures:
                    # Collect all percentage changes
                    all_pct_series = []
                    
                    # Determine the maximum data points (limited to 2x pattern length for consistency)
                    max_len = min(max([len(m['pct_changes']) for m in match_futures]), target_future_points)
                    
                    # Pad all series to the same length
                    for match_future in match_futures:
                        # Trim to max_len if needed (for consistency with other elements)
                        pct_series = pd.Series(match_future['pct_changes'][:max_len])
                        # Pad with NaN if needed
                        padded = pct_series.reindex(range(max_len), fill_value=np.nan)
                        all_pct_series.append(padded)
                    
                    # Stack into a DataFrame
                    pct_df = pd.concat(all_pct_series, axis=1)
                    
                    if use_weighted:
                        # Use weighted average based on match quality
                        match_scores = [m['distance'] for m in match_futures]  # Access distance directly
                        max_score = max(match_scores)
                        min_score = min(match_scores)
                        
                        # Ensure we don't divide by zero
                        score_range = max_score - min_score
                        if score_range == 0:
                            # Equal weights if all scores are the same
                            weights = [1.0 for _ in match_scores]
                        else:
                            # Calculate inverse weights (lower score = better match = higher weight)
                            weights = [(max_score - score) / score_range for score in match_scores]
                            
                            # Apply quadratic weighting to emphasize better matches more
                            weights = [w**2 for w in weights]
                            
                            # Normalize weights to sum to 1
                            sum_weights = sum(weights)
                            weights = [w / sum_weights for w in weights]
                        
                        # Apply weights to each column
                        weighted_df = pct_df.copy()
                        for i, col in enumerate(weighted_df.columns):
                            weighted_df[col] = weighted_df[col] * weights[i]
                            
                        # Sum across columns to get weighted average
                        weighted_pcts = weighted_df.sum(axis=1, skipna=True)
                        prediction_pcts = weighted_pcts
                        path_name = "Weighted Path"
                    else:
                        # Use standard median approach
                        prediction_pcts = pct_df.median(axis=1, skipna=True)
                        path_name = "Median Path"
                    
                    # Convert back to prices
                    prediction_prices = [reference_price * (1 + pct/100) for pct in prediction_pcts]
                    
                    # Make sure we have enough projection_times for the prediction
                    while len(projection_times) < len(prediction_prices):
                        next_time = reference_time + timedelta(seconds=avg_time_diff * (len(projection_times) + 1))
                        projection_times.append(next_time)
                    
                    # Add reference horizontal line at the reference price level
                    pred_fig.add_shape(
                        type="line",
                        x0=reference_time,  # Start at reference point
                        x1=projection_times[-1],  # End at last projection time point
                        y0=reference_price,  # At the reference price level
                        y1=reference_price,
                        line=dict(
                            color="rgba(150, 150, 150, 0.7)",
                            width=1.5,
                            dash="dot"
                        )
                    )
                    
                    # Add the median path with reduced thickness (2 instead of 3)
                    # Keep the original look and feel, just use prediction values
                    pred_fig.add_trace(
                        go.Scatter(
                            x=[reference_time] + projection_times[:len(prediction_prices)],
                            y=[reference_price] + prediction_prices,
                            mode='lines',
                            line=dict(
                                color='rgba(255, 255, 255, 1.0)',  # White
                                width=2,  # Reduced thickness from 3 to 2
                                dash='solid'  # Always use solid line, regardless of weighted or not
                            ),
                            name=path_name,
                            hoverinfo="y+text",
                            hovertext=["Reference"] + [f"{pct:.2f}% change" for pct in prediction_pcts.values]
                        )
                    )
                
                # NEW: Add actual price data if this is a historical pattern
                # Check if we have the main dataframe and indexes, and if there's data after the pattern
                has_actual_data = False
                actual_pct_changes = []
                actual_future_data = None
                
                if main_df is not None and start_idx is not None and end_idx is not None:
                    # Check if the pattern is not at the end of the data
                    if end_idx < len(main_df) - 1:
                        # Determine how many future points to show based on the pattern length
                        pattern_length = end_idx - start_idx + 1  # Number of candles in the pattern
                        
                        # For future prediction, we want to show 2x the pattern length
                        # This matches how we display historical patterns
                        target_future_length = pattern_length * 2
                        
                        # Calculate how many future points we can actually show
                        available_future_length = len(main_df) - end_idx - 1
                        future_length = min(target_future_length, available_future_length)
                        
                        if future_length > 0:
                            # Extract the actual future data
                            future_start_idx = end_idx + 1
                            future_end_idx = future_start_idx + future_length - 1
                            future_end_idx = min(future_end_idx, len(main_df) - 1)  # Ensure we don't go past the data
                            
                            actual_future_data = main_df.iloc[future_start_idx:future_end_idx+1]
                            
                            # Calculate percentage changes from reference point (last candle of pattern)
                            # for statistics calculation only
                            actual_reference_price = main_df.iloc[end_idx]['close']
                            actual_pct_changes = [(price / actual_reference_price - 1) * 100 for price in actual_future_data['close']]
                            
                            # Get key statistics about what actually happened after the pattern
                            actual_final_price = actual_future_data['close'].iloc[-1] if not actual_future_data.empty else 0
                            actual_final_pct_change = ((actual_final_price / actual_reference_price) - 1) * 100
                            
                            has_actual_data = True
                
                # Add the actual candles if available
                if has_actual_data and actual_future_data is not None and not actual_future_data.empty:
                    # First, let's add a reference point - a thin dotted line at the level of the last pattern close price
                    # This makes it easier to visually compare where the price started versus where it ended up
                    pattern_end_price = main_df.iloc[end_idx]['close']  # The actual closing price at end of pattern
                    
                    # For the visualization scaling, we need to make the ranges compatible
                    # Calculate the normalized prices relative to our reference price for consistency in the chart
                    reference_ratio = reference_price / pattern_end_price
                    
                    # Normalize all OHLC data relative to our reference price for chart display
                    # This is necessary for proper visualization - we're not changing the data, just scaling it 
                    # to appear correctly on the same chart with the pattern matches
                    normalized_open = [price * reference_ratio for price in actual_future_data['open']]
                    normalized_high = [price * reference_ratio for price in actual_future_data['high']]
                    normalized_low = [price * reference_ratio for price in actual_future_data['low']]
                    normalized_close = [price * reference_ratio for price in actual_future_data['close']]
                    
                    # Create a consistent time grid that ensures the actual data terminates at the same point
                    # as the match projections. We use the projection_times that were calculated earlier
                    # for the match futures.
                    
                    # Calculate exactly how many projection points we need (2x pattern length)
                    # (target_future_points is already defined above)
                    
                    # Ensure we have enough projection times for 2x pattern length
                    while len(projection_times) < target_future_points:
                        next_time = reference_time + timedelta(seconds=avg_time_diff * (len(projection_times) + 1))
                        projection_times.append(next_time)
                    
                    # Extract just the number of projection times we need
                    aligned_times = projection_times[:target_future_points]
                    
                    # If we have more actual data points than projection points, truncate
                    # If we have fewer, we'll use only the times we have data for
                    data_points = min(len(actual_future_data), len(aligned_times))
                    
                    # Add reference horizontal line at the pattern end price level (normalized)
                    # Use the last projection time to ensure the line extends to the same end point as projections
                    pred_fig.add_shape(
                        type="line",
                        x0=reference_time,  # Start at reference point
                        x1=aligned_times[-1],  # End at last projection time point, ensuring same end point as projections
                        y0=reference_price,  # At the reference price level
                        y1=reference_price,
                        line=dict(
                            color="rgba(150, 150, 150, 0.7)",
                            width=1.5,
                            dash="dot"
                        )
                    )
                    
                    # Add actual future data as candlesticks with standard styling
                    # Only use the data points we have, but align them to our projection time grid
                    pred_fig.add_trace(
                        go.Candlestick(
                            x=aligned_times[:data_points],  # Use aligned times instead of actual datetime
                            open=normalized_open[:data_points],
                            high=normalized_high[:data_points],
                            low=normalized_low[:data_points],
                            close=normalized_close[:data_points],
                            name="Actual Outcome",
                            increasing=dict(
                                line=dict(color=style['increasing_line_color']),
                                fillcolor=style['increasing_color']
                            ),
                            decreasing=dict(
                                line=dict(color=style['decreasing_line_color']),
                                fillcolor=style['decreasing_color']
                            ),
                            hoverinfo="x+y"
                        )
                    )
                    
                    # For stats, use the actual percentage change from the final candle
                    final_actual_pct = actual_final_pct_change
                
                # Calculate y-axis range to include all data points plus padding
                all_prices = [reference_price]
                for match_future in match_futures:
                    min_price = reference_price * (1 + match_future['min_pct']/100)
                    max_price = reference_price * (1 + match_future['max_pct']/100)
                    all_prices.extend([min_price, max_price])
                
                # Include actual OHLC prices in the range calculation if available
                if has_actual_data and actual_future_data is not None and not actual_future_data.empty:
                    reference_ratio = reference_price / main_df.iloc[end_idx]['close']
                    normalized_highs = [price * reference_ratio for price in actual_future_data['high']]
                    normalized_lows = [price * reference_ratio for price in actual_future_data['low']]
                    all_prices.extend(normalized_highs)
                    all_prices.extend(normalized_lows)
                
                y_min = min(all_prices) * 0.99  # 1% padding below
                y_max = max(all_prices) * 1.01  # 1% padding above
                
                # Define explicit x-axis range to ensure everything is properly aligned and bounded
                x_min = reference_time
                
                # Determine pattern length for calculating target future range
                pattern_length = 0
                if start_idx is not None and end_idx is not None:
                    pattern_length = end_idx - start_idx + 1
                
                # Calculate how far into the future we should display (2x pattern length)
                target_future_points = pattern_length * 2
                
                # For the max range, use a consistent target future length (2x pattern length)
                # This ensures the actual and predicted data are displayed with the same scale
                if avg_time_diff > 0:
                    # Set the view to exactly 2x pattern length into the future
                    # This is the target end point for ALL visualization elements (match projections, median, real data)
                    x_max = reference_time + timedelta(seconds=avg_time_diff * target_future_points)
                
                # Add 5% padding on the x-axis
                time_range = x_max - x_min
                padding = time_range * 0.05
                x_min_padded = x_min - padding
                x_max_padded = x_max + padding
                
                # Configure layout with enhanced hover effects
                pred_fig.update_layout(
                    title=None,  # Remove chart title since we have styled title above
                    height=690,  # Match height with expanded view charts
                    xaxis_rangeslider_visible=False,
                    plot_bgcolor=style['background_color'],
                    paper_bgcolor=style['background_color'],
                    font=dict(color='white', family="ProtoMono-Light, monospace"),
                    margin=dict(l=25, r=25, t=0, b=15),  # Removed top margin completely to match match charts
                    showlegend=False,
                    hoverlabel=dict(
                        bgcolor=style['background_color'],
                        font_size=14,
                        font_family="ProtoMono-Light, monospace"
                    ),
                    shapes=[],
                    annotations=[],
                    xaxis_showticklabels=True,
                    yaxis_showticklabels=True,
                    modebar_remove=["lasso", "select"],
                    # Use more focused hover that only displays when directly over points
                    hovermode='closest',
                    hoverdistance=10  # Only show hover info when very close to points
                    # Removed explicit range setting to allow Plotly's autorange to work properly
                )
                
                # This section is redundant - we already defined these variables earlier
                # Ensure projection_times has enough points to reach x_max
                while len(projection_times) < target_future_points:
                    next_time = reference_time + timedelta(seconds=avg_time_diff * (len(projection_times) + 1))
                    projection_times.append(next_time)
                    
                # Add invisible "anchor points" at the extremes to help Plotly's autorange show the complete chart
                # Use 5% padding on left but only 1% on right as requested
                padding_left = time_range * 0.05  # 5% padding on left
                padding_right = time_range * 0.01  # 1% padding on right
                x_min_wide = x_min - padding_left
                x_max_wide = x_max + padding_right
                
                # Add invisible anchor point at the far left
                pred_fig.add_trace(
                    go.Scatter(
                        x=[x_min_wide],
                        y=[reference_price],  # Use reference price level
                        mode='markers',
                        marker=dict(size=0, opacity=0),  # Completely invisible
                        showlegend=False,
                        hoverinfo='none'
                    )
                )
                
                # Add invisible anchor point at the far right
                pred_fig.add_trace(
                    go.Scatter(
                        x=[x_max_wide],
                        y=[reference_price],  # Use reference price level
                        mode='markers',
                        marker=dict(size=0, opacity=0),  # Completely invisible
                        showlegend=False,
                        hoverinfo='none'
                    )
                )
                
                # Update x-axes with grid lines and flat time labels, without hover spikes
                # Allow Plotly to autorange the x-axis based on all data points and anchor points
                pred_fig.update_xaxes(
                    showgrid=True,
                    gridcolor=style['grid_color'],
                    gridwidth=0.5,
                    zeroline=False,
                    showticklabels=True,
                    linecolor=style['grid_color'],
                    tickangle=0,  # Flat time labels
                    tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),
                    tickformat="%m-%d<br>%H:%M",  # Two-line format with month-day on top, hours below
                    nticks=15,
                    ticks="outside",  # Place ticks outside the chart
                    ticklen=8,  # Longer tick marks
                    minor_showgrid=True,  # Show minor grid lines too
                    minor_gridcolor=style['grid_color'],
                    tickcolor="#999999",
                    # Remove hover spikes
                    showspikes=False
                )
                
                # Set different tick spacing based on selected coin for y-axis
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
                    
                # Update y-axis with appropriate range and grid, without hover spikes
                pred_fig.update_yaxes(
                    range=[y_min, y_max],
                    showgrid=True,
                    gridcolor=style['grid_color'],
                    gridwidth=0.5,
                    zeroline=False,
                    linecolor=style['grid_color'],
                    dtick=tick_spacing,
                    tickmode="linear",
                    tick0=0,
                    tickformat=",.0f",
                    tickformatstops=tickformatstops_config,
                    tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),
                    tickcolor="#999999",
                    hoverformat=",.0f",
                    showline=False,
                    mirror=False,
                    # Remove hover spikes
                    showspikes=False
                )
                
                # Update statistics - use weighted average if use_weighted is True
                if use_weighted and match_futures:
                    # Extract match scores and calculate weights (same as for the prediction line)
                    match_scores = [m['distance'] for m in match_futures]
                    max_score = max(match_scores)
                    min_score = min(match_scores)
                    
                    # Ensure we don't divide by zero
                    score_range = max_score - min_score
                    if score_range == 0:
                        # Equal weights if all scores are the same
                        weights = [1.0 for _ in match_scores]
                    else:
                        # Calculate inverse weights (lower score = better match = higher weight)
                        weights = [(max_score - score) / score_range for score in match_scores]
                        
                        # Apply quadratic weighting to emphasize better matches more
                        weights = [w**2 for w in weights]
                        
                        # Normalize weights to sum to 1
                        sum_weights = sum(weights)
                        weights = [w / sum_weights for w in weights]
                    
                    # Calculate weighted statistics
                    weighted_min = sum(min_pct * weight for min_pct, weight in zip(min_pct_changes, weights)) if min_pct_changes else 0
                    weighted_max = sum(max_pct * weight for max_pct, weight in zip(max_pct_changes, weights)) if max_pct_changes else 0
                    weighted_final = sum(final_pct * weight for final_pct, weight in zip(final_pct_changes, weights)) if final_pct_changes else 0
                    
                    stats = {
                        'num_matches': len(match_futures),
                        'avg_min': weighted_min,
                        'avg_max': weighted_max,
                        'avg_final': weighted_final,
                        'has_actual_data': has_actual_data,
                        'actual_final_pct': final_actual_pct if has_actual_data else None
                    }
                else:
                    # Use simple average (mean) for all stats when not using weighted mode
                    stats = {
                        'num_matches': len(match_futures),
                        'avg_min': np.mean(min_pct_changes) if min_pct_changes else 0,
                        'avg_max': np.mean(max_pct_changes) if max_pct_changes else 0,
                        'avg_final': np.mean(final_pct_changes) if final_pct_changes else 0,
                        'has_actual_data': has_actual_data,
                        'actual_final_pct': final_actual_pct if has_actual_data else None
                    }
                
                return pred_fig, stats
            
            # Create and display prediction chart before showing individual matches
            if filtered_matches and len(filtered_matches) > 0:
                # Create the prediction visualization
                result = create_prediction_chart(
                    source_pattern, 
                    filtered_matches, 
                    style=st.session_state.candle_style,
                    max_matches=len(filtered_matches),  # Use all filtered matches instead of hardcoded limit
                    main_df=df,  # Pass main dataframe to extract actual future data
                    start_idx=start_idx,  # Pass start index for the pattern
                    end_idx=end_idx,  # Pass end index for the pattern
                    use_weighted=use_weighted  # Pass weighted option
                )
                
                # Unpack return value - contains figure and stats
                pred_fig, stats = result
                
                # Display the chart with a consistent title style
                if pred_fig is not None:
                    # Let's use the same approach as the Match titles - no divs or custom CSS
                    # This simpler approach will ensure spacing is identical to match titles
                    # Add indication if actual data is shown
                    # Add 'Weighted' to the title when weighted prediction is used
                    stat_prefix = "Weighted" if use_weighted else "Avg"
                    title_text = f"**Prediction: Pattern Outcomes** (Based on {stats['num_matches']} matches - {stat_prefix} Min: {stats['avg_min']:.2f}% | {stat_prefix} Max: {stats['avg_max']:.2f}% | {stat_prefix} Final: {stats['avg_final']:.2f}%)"
                    
                    # If we have actual data, add it to the title - keep it in the standard text color
                    if stats.get('has_actual_data', False) and stats.get('actual_final_pct') is not None:
                        title_text += f" | Actual Final: {stats['actual_final_pct']:.2f}%"
                        
                    st.markdown(title_text, unsafe_allow_html=True)
                    
                    # No JavaScript needed, using simple static visualization
                    
                    # Display chart without title (title is now in markdown)
                    # Include the same config as the main chart to ensure consistent behavior
                    chart_config = {
                        'scrollZoom': True,
                        'displaylogo': False,
                    }
                    st.plotly_chart(pred_fig, use_container_width=True, config=chart_config)
            
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
                            fillcolor='rgba(66, 135, 245, 0.7)'          # Light blue with transparency
                        ),
                        decreasing=dict(
                            line=dict(color='rgba(26, 86, 196, 0.7)'),   # Darker blue
                            fillcolor='rgba(26, 86, 196, 0.7)'           # Darker blue with transparency
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
            # Use consistent chart configuration
            chart_config = {
                'scrollZoom': True,
                'displaylogo': False,
            }
            st.plotly_chart(source_fig, use_container_width=True, config=chart_config)
            
            # Add a small spacer before the charts
            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            
            # Function to create expanded view chart
            def create_expanded_match_view(source_data, match_data, pattern_indices, match_date):
                """Create expanded charts for both match and reference pattern"""
                
                # Calculate how much context to show (equal on both sides)
                match_start, match_end = pattern_indices
                pattern_length = match_end - match_start + 1
                context_size = pattern_length * 2  # Show 2x pattern length as context
                
                # Get all indices as a list for easier manipulation
                all_indices = list(range(len(match_data)))
                                
                # Create figure with two subplots stacked vertically
                fig = make_subplots(
                    rows=2, 
                    cols=1,
                    # Create regime-colored titles with HTML
                    subplot_titles=(
                        f"Matched Pattern from {match_date}",
                        f"Reference Pattern"
                    ),
                    vertical_spacing=0.12,
                    row_heights=[0.5, 0.5]  # Equal height for both charts
                )
                
                # 1. Add matched pattern with context (all data with regular colors)
                fig.add_trace(
                    go.Candlestick(
                        x=match_data.index,
                        open=match_data['open'],
                        high=match_data['high'],
                        low=match_data['low'],
                        close=match_data['close'],
                        increasing_line_color='#26a69a', 
                        decreasing_line_color='#ef5350',
                        name="Context"
                    ),
                    row=1, col=1
                )
                
                # 2. Highlight the actual pattern in blue (overlay)
                # Only if the pattern indices are valid
                if match_start < len(match_data) and match_end < len(match_data):
                    pattern_data = match_data.iloc[match_start:match_end+1]
                    fig.add_trace(
                        go.Candlestick(
                            x=pattern_data.index,
                            open=pattern_data['open'],
                            high=pattern_data['high'],
                            low=pattern_data['low'],
                            close=pattern_data['close'],
                            increasing_line_color='#1E88E5', 
                            decreasing_line_color='#1E88E5',
                            name="Pattern Match"
                        ),
                        row=1, col=1
                    )
                
                    # 3. Add vertical line at the end of the pattern
                    if match_end < len(match_data):
                        fig.add_vline(
                            x=match_data.index[match_end], 
                            line_width=2, 
                            line_dash="dash", 
                            line_color="#1E88E5",
                            row=1
                        )
                
                # 4. Add reference pattern (source pattern with context)
                fig.add_trace(
                    go.Candlestick(
                        x=source_data.index,
                        open=source_data['open'],
                        high=source_data['high'],
                        low=source_data['low'],
                        close=source_data['close'],
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350',
                        name="Reference Context"
                    ),
                    row=2, col=1
                )
                
                # 5. Highlight the pattern part of the reference (blue overlay)
                # We assume the pattern is at the end of the source data
                pattern_start = max(0, len(source_data) - pattern_length)
                pattern_data = source_data.iloc[pattern_start:]
                
                fig.add_trace(
                    go.Candlestick(
                        x=pattern_data.index,
                        open=pattern_data['open'],
                        high=pattern_data['high'],
                        low=pattern_data['low'],
                        close=pattern_data['close'],
                        increasing_line_color='#1E88E5',
                        decreasing_line_color='#1E88E5',
                        name="Reference Pattern"
                    ),
                    row=2, col=1
                )
                
                # Configure layout to match main chart style
                # Get regime info
                match_regime = match.get('regime', 3)
                match_regime_name = match.get('regime_name', 'Neutral')
                source_regime = results.get('source_regime', 3)
                source_regime_name = results.get('source_regime_name', 'Neutral')
                
                # Define regime colors
                regime_colors = {
                    1: "#4CAF50",  # Bullish-Stable: Green
                    2: "#8BC34A",  # Bullish-Volatile: Light Green
                    3: "#9E9E9E",  # Neutral: Grey
                    4: "#FF9800",  # Bearish-Stable: Orange
                    5: "#F44336",  # Bearish-Volatile: Red
                    6: "#2196F3"   # Choppy: Blue
                }
                
                # Get colors for regimes
                match_color = regime_colors.get(match_regime, "#9E9E9E")
                source_color = regime_colors.get(source_regime, "#9E9E9E")
                
                fig.update_layout(
                    height=800,  # Tall chart for detail
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=50, b=10),
                    showlegend=False,
                    xaxis_rangeslider_visible=False,
                    xaxis2_rangeslider_visible=False
                )
                
                # Add annotations for regime info
                fig.add_annotation(
                    text=f"Regime: {match_regime_name}",
                    xref="paper", yref="paper",
                    x=0.5, y=1.0,
                    showarrow=False,
                    font=dict(color=match_color, size=14),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor=match_color,
                    borderwidth=1,
                    borderpad=4,
                    row=1, col=1
                )
                
                fig.add_annotation(
                    text=f"Regime: {source_regime_name}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.45,
                    showarrow=False,
                    font=dict(color=source_color, size=14),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor=source_color,
                    borderwidth=1,
                    borderpad=4,
                    row=2, col=1
                )
                
                return fig
            
            # Precompute the final price comparison for each match
            match_final_results = {}
            for match_idx, match in enumerate(filtered_matches):
                match_data = pd.DataFrame(match["pattern_data"])
                
                # Only analyze if we have enough data after the pattern
                if len(match_data) > pattern_length:
                    # Get the last close price of the pattern part (before the blue line)
                    pattern_last_close = match_data.iloc[pattern_length-1]['close']
                    
                    # Get the last close price of the future part (after the blue line)
                    future_last_close = match_data.iloc[-1]['close']
                    
                    # Store comparison result for this match
                    if future_last_close > pattern_last_close:
                        match_final_results[match_idx] = "Higher"
                    elif future_last_close < pattern_last_close:
                        match_final_results[match_idx] = "Lower"
                    else:
                        match_final_results[match_idx] = "Unchanged"
                else:
                    match_final_results[match_idx] = "N/A"  # Not enough future data
            
            # Show all matches directly
            for i, match in enumerate(filtered_matches):
                # Get the final price comparison result for this match
                final_result = match_final_results.get(i, "N/A")
                
                # Style the result with color
                if final_result == "Higher":
                    final_result_styled = f"<span style='color:#4CAF50'>Higher</span>"  # Green
                elif final_result == "Lower":
                    final_result_styled = f"<span style='color:#F44336'>Lower</span>"   # Red
                else:
                    final_result_styled = f"<span style='color:#999999'>{final_result}</span>"  # Grey
                
                # Create columns for match header - title on left, button on right
                st.markdown('<div class="match_header_cols">', unsafe_allow_html=True)
                match_header_cols = st.columns([0.85, 0.15])
                
                # Header with match info, regime and median comparison 
                score_label = "Feature Distance" if results['type'] == 'feature_pattern' else "Shape Distance"
                
                # Get regime info with appropriate styling based on regime
                match_regime = match.get('regime', 3)  # Default to neutral if not available
                match_regime_name = match.get('regime_name', 'Neutral')
                
                # Style regimes with colors
                regime_colors = {
                    1: "#4CAF50",  # Bullish-Stable: Green
                    2: "#8BC34A",  # Bullish-Volatile: Light Green
                    3: "#9E9E9E",  # Neutral: Grey
                    4: "#FF9800",  # Bearish-Stable: Orange
                    5: "#F44336",  # Bearish-Volatile: Red
                    6: "#2196F3"   # Choppy: Blue
                }
                regime_color = regime_colors.get(match_regime, "#9E9E9E")
                regime_styled = f"<span style='color:{regime_color}'>{match_regime_name}</span>"
                
                with match_header_cols[0]:
                    st.markdown(
                        f"**Match #{i+1}**: {datetime.fromisoformat(match['start_time'].replace('Z', '')).strftime('%Y-%m-%d')} "
                        f"({score_label}: {match['distance']:.4f}). Regime: {regime_styled}. Final price: {final_result_styled}", 
                        unsafe_allow_html=True
                    )
                
                # Add Expand View button on the right
                with match_header_cols[1]:
                    # Initialize expanded state for this match if not already in session state
                    if f"expand_state_{i}" not in st.session_state:
                        st.session_state[f"expand_state_{i}"] = False
                    
                    # No need for per-button custom CSS anymore
                    
                    # Button to toggle expanded view
                    button_label = "Collapse" if st.session_state[f"expand_state_{i}"] else "Expand"
                    if st.button(button_label, key=f"expand_button_{i}", type="secondary"):
                        # Toggle the expanded state for this match
                        st.session_state[f"expand_state_{i}"] = not st.session_state[f"expand_state_{i}"]
                
                # Close the match header div
                st.markdown('</div>', unsafe_allow_html=True)
                
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
                # Use consistent chart configuration
                chart_config = {
                    'scrollZoom': True,
                    'displaylogo': False,
                }
                st.plotly_chart(match_fig, use_container_width=True, config=chart_config)
                
                # If expanded view is toggled for this match, display expanded view right after this chart
                if st.session_state.get(f"expand_state_{i}", False):
                    with st.container():
                        # Add a small visual separator
                        st.markdown("<div style='border-left: 4px solid #4287f5; padding-left: 10px; margin: 10px 0;'>", unsafe_allow_html=True)
                        
                        # Get the pattern length and match date
                        pattern_length = len(source_pattern)
                        match_date = datetime.fromisoformat(match['start_time'].replace('Z', '')).strftime('%Y-%m-%d')
                        match_df = pd.DataFrame(match["pattern_data"])
                        pattern_indices = (0, pattern_length-1)
                        
                        # Create match chart with full context
                        st.write("**Match Pattern in Full Context**")
                        
                        # Get data in datetime format
                        match_df_dates = match_df.copy()
                        match_df_dates['datetime'] = [datetime.fromtimestamp(ts/1000) for ts in match_df_dates['timestamp']]
                        match_start, match_end = pattern_indices
                        
                        # Create the context data and charts
                        # Load data provider to get more context around the match
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        provider = get_data_provider()
                        
                        # Define the pattern section - use first pattern_length candles to match normal view
                        pattern_section = match_df_dates.iloc[:pattern_length]
                        pattern_midpoint = pattern_section['datetime'].mean()
                        
                        # Calculate time range for context - use the same range as the main chart
                        days_to_load = st.session_state.days_to_load if 'days_to_load' in st.session_state else 30
                        interval = st.session_state.interval if 'interval' in st.session_state else '30m'
                        
                        pattern_midpoint_ts = pattern_midpoint.timestamp() * 1000  # Convert to milliseconds
                        half_range_ms = days_to_load * 24 * 60 * 60 * 1000 / 2  # Half the range in milliseconds
                        context_start_ts = pattern_midpoint_ts - half_range_ms
                        context_end_ts = pattern_midpoint_ts + half_range_ms
                        
                        # Convert to ISO format for the API
                        context_start_time = datetime.fromtimestamp(context_start_ts/1000).isoformat()
                        context_end_time = datetime.fromtimestamp(context_end_ts/1000).isoformat()
                        
                        # Create expanded view match figure
                        match_exp_fig = go.Figure()
                        
                        try:
                            # Get full context data
                            context_data = loop.run_until_complete(
                                provider.get_historical_ohlcv(
                                    symbol=st.session_state.selected_coin,
                                    interval=interval,
                                    start_time=context_start_time,
                                    end_time=context_end_time
                                )
                            )
                            
                            if context_data is not None and not context_data.empty:
                                # Convert timestamp to datetime for plotting
                                context_data['datetime'] = pd.to_datetime(context_data['timestamp'], unit='ms')
                                
                                # Find where our pattern is within this context
                                pattern_start_time = pattern_section['datetime'].min()
                                pattern_end_time = pattern_section['datetime'].max()
                                
                                # Plot the full context with regular candle colors
                                match_exp_fig.add_trace(
                                    go.Candlestick(
                                        x=context_data['datetime'],
                                        open=context_data['open'],
                                        high=context_data['high'],
                                        low=context_data['low'],
                                        close=context_data['close'],
                                        increasing_line_color=st.session_state.candle_style['increasing_line_color'],
                                        decreasing_line_color=st.session_state.candle_style['decreasing_line_color'],
                                        increasing_fillcolor=st.session_state.candle_style['increasing_color'],
                                        decreasing_fillcolor=st.session_state.candle_style['decreasing_color'],
                                        name="Context"
                                    )
                                )
                                
                                # Get exact timestamps from original match data for precise matching
                                pattern_timestamps = match_df_dates.iloc[:pattern_length]['timestamp'].tolist()
                                
                                # Find the indices in context_data that exactly match our pattern timestamps
                                pattern_in_context = context_data[context_data['timestamp'].isin(pattern_timestamps)]
                                
                                if not pattern_in_context.empty:
                                    # Overlay the pattern with blue candles
                                    match_exp_fig.add_trace(
                                        go.Candlestick(
                                            x=pattern_in_context['datetime'],
                                            open=pattern_in_context['open'],
                                            high=pattern_in_context['high'],
                                            low=pattern_in_context['low'],
                                            close=pattern_in_context['close'],
                                            increasing_line_color='rgba(66, 135, 245, 0.7)',  # Light blue
                                            decreasing_line_color='rgba(26, 86, 196, 0.7)',   # Darker blue
                                            name="Pattern Match"
                                        )
                                    )
                                    
                                    # Removed vertical line as requested
                            else:
                                # Fallback to using just the match data if we can't get context
                                match_exp_fig.add_trace(
                                    go.Candlestick(
                                        x=match_df_dates['datetime'],
                                        open=match_df_dates['open'],
                                        high=match_df_dates['high'],
                                        low=match_df_dates['low'],
                                        close=match_df_dates['close'],
                                        increasing_line_color=st.session_state.candle_style['increasing_line_color'],
                                        decreasing_line_color=st.session_state.candle_style['decreasing_line_color'],
                                        increasing_fillcolor=st.session_state.candle_style['increasing_color'],
                                        decreasing_fillcolor=st.session_state.candle_style['decreasing_color'],
                                        name="Context"
                                    )
                                )
                                
                                # Overlay pattern section with blue candles
                                match_exp_fig.add_trace(
                                    go.Candlestick(
                                        x=pattern_section['datetime'],
                                        open=pattern_section['open'],
                                        high=pattern_section['high'],
                                        low=pattern_section['low'],
                                        close=pattern_section['close'],
                                        increasing_line_color='rgba(66, 135, 245, 0.7)',  # Light blue
                                        decreasing_line_color='rgba(26, 86, 196, 0.7)',   # Darker blue
                                        name="Pattern Match"
                                    )
                                )
                                
                        except Exception as e:
                            st.error(f"Error loading context data: {str(e)}")
                            # Fallback to just showing what we have
                            match_exp_fig.add_trace(
                                go.Candlestick(
                                    x=match_df_dates['datetime'],
                                    open=match_df_dates['open'],
                                    high=match_df_dates['high'],
                                    low=match_df_dates['low'],
                                    close=match_df_dates['close'],
                                    increasing_line_color=st.session_state.candle_style['increasing_line_color'],
                                    decreasing_line_color=st.session_state.candle_style['decreasing_line_color'],
                                    name="Match Pattern"
                                )
                            )
                            
                            # Overlay the pattern part in blue - using the first pattern_length rows
                            pattern_part = match_df_dates.iloc[:pattern_length]
                            match_exp_fig.add_trace(
                                go.Candlestick(
                                    x=pattern_part['datetime'],
                                    open=pattern_part['open'],
                                    high=pattern_part['high'],
                                    low=pattern_part['low'],
                                    close=pattern_part['close'],
                                    increasing_line_color='rgba(66, 135, 245, 0.7)',  # Light blue
                                    decreasing_line_color='rgba(26, 86, 196, 0.7)',   # Darker blue
                                    name="Pattern Match"
                                )
                            )
                        
                        # Configure the match chart to match main chart style exactly
                        match_exp_fig.update_layout(
                            height=690,  # Same as main chart
                            xaxis_rangeslider_visible=False,
                            plot_bgcolor=st.session_state.candle_style['background_color'],
                            paper_bgcolor=st.session_state.candle_style['background_color'],
                            font=dict(color='white', family="ProtoMono-Light, monospace"),
                            margin=dict(l=25, r=25, t=15, b=15),
                            hovermode='x unified',
                            showlegend=False,
                            hoverlabel=dict(
                                bgcolor=st.session_state.candle_style['background_color'],
                                font_size=14,
                                font_family="ProtoMono-Light, monospace"
                            ),
                            shapes=[],
                            annotations=[],
                            xaxis_showticklabels=True,
                            yaxis_showticklabels=True,
                            xaxis_showspikes=False,
                            yaxis_showspikes=False,
                            modebar_remove=["lasso", "select"],
                            title=None  # Remove the title
                        )
                        
                        # Update x-axes with grid lines and flat time labels
                        match_exp_fig.update_xaxes(
                            showgrid=True,
                            gridcolor=st.session_state.candle_style['grid_color'],
                            gridwidth=0.5,
                            zeroline=False,
                            showticklabels=True,
                            linecolor=st.session_state.candle_style['grid_color'],
                            tickangle=0,  # Flat time labels
                            tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),
                            tickformat="%m-%d<br>%H:%M",  # Two-line format with month-day on top, hours below
                            tickmode="auto",
                            nticks=15,  # Control the number of ticks
                            ticks="outside",  # Place ticks outside the chart
                            ticklen=8,  # Longer tick marks
                            minor_showgrid=True,  # Show minor grid lines too
                            minor_gridcolor=st.session_state.candle_style['grid_color'],
                            tickcolor="#999999"
                        )
                        
                        # Use dynamic tick spacing for y-axis based on selected coin
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
                            
                        match_exp_fig.update_yaxes(
                            showgrid=True,
                            gridcolor=st.session_state.candle_style['grid_color'],
                            gridwidth=0.5,
                            zeroline=False,
                            linecolor=st.session_state.candle_style['grid_color'],
                            dtick=tick_spacing,
                            tickmode="linear",
                            tick0=0,
                            tickformat=",.0f",
                            tickformatstops=tickformatstops_config,
                            tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),
                            tickcolor="#999999",
                            hoverformat=",.0f",
                            showline=False,
                            mirror=False
                        )
                        
                        # Display match chart with a unique key
                        # Use consistent chart configuration
                        chart_config = {
                            'scrollZoom': True,
                            'displaylogo': False,
                        }
                        st.plotly_chart(match_exp_fig, use_container_width=True, config=chart_config, key=f"match_exp_chart_{i}")
                        
                        # Create reference pattern chart
                        st.write("**Reference Pattern in Full Context**")
                        source_exp_fig = go.Figure()
                        
                        # We need to get the main chart data which contains the reference pattern
                        if df is not None and not df.empty:
                            # This is the main dataframe shown in the candlestick chart
                            # Get the reference pattern part using the selected range
                            start_idx = st.session_state.selected_range['start_idx']
                            end_idx = st.session_state.selected_range['end_idx']
                            
                            if start_idx is not None and end_idx is not None and start_idx < len(df) and end_idx < len(df):
                                # Plot full dataframe with regular colors
                                source_exp_fig.add_trace(
                                    go.Candlestick(
                                        x=df['datetime'],
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close'],
                                        increasing_line_color=st.session_state.candle_style['increasing_line_color'],
                                        decreasing_line_color=st.session_state.candle_style['decreasing_line_color'],
                                        increasing_fillcolor=st.session_state.candle_style['increasing_color'],
                                        decreasing_fillcolor=st.session_state.candle_style['decreasing_color'],
                                        name="Context"
                                    )
                                )
                                
                                # Overlay the selected pattern with blue colors
                                source_exp_fig.add_trace(
                                    go.Candlestick(
                                        x=df.iloc[start_idx:end_idx+1]['datetime'],
                                        open=df.iloc[start_idx:end_idx+1]['open'],
                                        high=df.iloc[start_idx:end_idx+1]['high'],
                                        low=df.iloc[start_idx:end_idx+1]['low'],
                                        close=df.iloc[start_idx:end_idx+1]['close'],
                                        increasing_line_color='rgba(66, 135, 245, 0.7)',  # Light blue
                                        decreasing_line_color='rgba(26, 86, 196, 0.7)',   # Darker blue
                                        name="Reference Pattern"
                                    )
                                )
                                
                                # Removed vertical line as requested
                            else:
                                # Fallback to source pattern from match
                                # Convert source pattern to DataFrame if it's not already
                                if isinstance(source_pattern, pd.DataFrame):
                                    source_df = source_pattern.copy()
                                else:
                                    source_df = pd.DataFrame(source_pattern)
                                
                                # Create datetime values for plotting
                                source_df['datetime'] = [datetime.fromtimestamp(ts/1000) for ts in source_df['timestamp']]
                                
                                # Add source pattern as blue candles
                                source_exp_fig.add_trace(
                                    go.Candlestick(
                                        x=source_df['datetime'],
                                        open=source_df['open'],
                                        high=source_df['high'],
                                        low=source_df['low'],
                                        close=source_df['close'],
                                        increasing_line_color='rgba(66, 135, 245, 0.7)',  # Light blue
                                        decreasing_line_color='rgba(26, 86, 196, 0.7)',   # Darker blue
                                        name="Reference Pattern"
                                    )
                                )
                        else:
                            # Fallback to source pattern from match
                            # Convert source pattern to DataFrame if it's not already
                            if isinstance(source_pattern, pd.DataFrame):
                                source_df = source_pattern.copy()
                            else:
                                source_df = pd.DataFrame(source_pattern)
                            
                            # Create datetime values for plotting
                            source_df['datetime'] = [datetime.fromtimestamp(ts/1000) for ts in source_df['timestamp']]
                            
                            # Add source pattern as blue candles
                            source_exp_fig.add_trace(
                                go.Candlestick(
                                    x=source_df['datetime'],
                                    open=source_df['open'],
                                    high=source_df['high'],
                                    low=source_df['low'],
                                    close=source_df['close'],
                                    increasing_line_color='rgba(66, 135, 245, 0.7)',  # Light blue
                                    decreasing_line_color='rgba(26, 86, 196, 0.7)',   # Darker blue
                                    name="Reference Pattern"
                                )
                            )
                        
                        # Configure the reference chart to match main chart style exactly
                        source_exp_fig.update_layout(
                            height=690,  # Same as main chart
                            xaxis_rangeslider_visible=False,
                            plot_bgcolor=st.session_state.candle_style['background_color'],
                            paper_bgcolor=st.session_state.candle_style['background_color'],
                            font=dict(color='white', family="ProtoMono-Light, monospace"),
                            margin=dict(l=25, r=25, t=15, b=15),
                            hovermode='x unified',
                            showlegend=False,
                            hoverlabel=dict(
                                bgcolor=st.session_state.candle_style['background_color'],
                                font_size=14,
                                font_family="ProtoMono-Light, monospace"
                            ),
                            shapes=[],
                            annotations=[],
                            xaxis_showticklabels=True,
                            yaxis_showticklabels=True,
                            xaxis_showspikes=False,
                            yaxis_showspikes=False,
                            modebar_remove=["lasso", "select"],
                            title=None  # Remove the title
                        )
                        
                        # Update x-axes with grid lines and flat time labels
                        source_exp_fig.update_xaxes(
                            showgrid=True,
                            gridcolor=st.session_state.candle_style['grid_color'],
                            gridwidth=0.5,
                            zeroline=False,
                            showticklabels=True,
                            linecolor=st.session_state.candle_style['grid_color'],
                            tickangle=0,  # Flat time labels
                            tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),
                            tickformat="%m-%d<br>%H:%M",  # Two-line format with month-day on top, hours below
                            tickmode="auto",
                            nticks=15,  # Control the number of ticks
                            ticks="outside",  # Place ticks outside the chart
                            ticklen=8,  # Longer tick marks
                            minor_showgrid=True,  # Show minor grid lines too
                            minor_gridcolor=st.session_state.candle_style['grid_color'],
                            tickcolor="#999999"
                        )
                        
                        # Use dynamic tick spacing for y-axis based on selected coin
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
                            
                        source_exp_fig.update_yaxes(
                            showgrid=True,
                            gridcolor=st.session_state.candle_style['grid_color'],
                            gridwidth=0.5,
                            zeroline=False,
                            linecolor=st.session_state.candle_style['grid_color'],
                            dtick=tick_spacing,
                            tickmode="linear",
                            tick0=0,
                            tickformat=",.0f",
                            tickformatstops=tickformatstops_config,
                            tickfont=dict(family="ProtoMono-Light, monospace", color="#999999"),
                            tickcolor="#999999",
                            hoverformat=",.0f",
                            showline=False,
                            mirror=False
                        )
                        
                        # Display reference chart with a unique key
                        # Use consistent chart configuration
                        chart_config = {
                            'scrollZoom': True,
                            'displaylogo': False,
                        }
                        st.plotly_chart(source_exp_fig, use_container_width=True, config=chart_config, key=f"source_exp_chart_{i}")
                        
                        # Add a close button
                        if st.button("Close Expanded View", key=f"close_expand_{i}", use_container_width=True):
                            st.session_state[f"expand_state_{i}"] = False
                            st.rerun()
                        
                        # Close the container div
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No matching patterns found.")
            
# Clean up old modal variables if present
if 'show_modal' in st.session_state:
    del st.session_state['show_modal']
if 'modal_match_id' in st.session_state:
    del st.session_state['modal_match_id']
if 'modal_match_date' in st.session_state:
    del st.session_state['modal_match_date']
if 'modal_match_df' in st.session_state:
    del st.session_state['modal_match_df']
if 'modal_distance' in st.session_state:
    del st.session_state['modal_distance']
if 'modal_pattern_indices' in st.session_state:
    del st.session_state['modal_pattern_indices']
if 'modal_median_result' in st.session_state:
    del st.session_state['modal_median_result']