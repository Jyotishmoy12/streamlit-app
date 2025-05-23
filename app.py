import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64
from PIL import Image
import json
import time
import os
import glob
from pathlib import Path

# ADD THIS CODE HERE - RIGHT AFTER IMPORTS
# Detect environment

# Page configuration (your existing code continues here)
st.set_page_config(
    page_title="Leaf Health Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

IS_CLOUD = not os.path.exists(os.path.expanduser("~/Desktop"))

if IS_CLOUD:
    st.warning("⚠️ Cloud Mode - Upload files manually")
    # Add file uploaders for CSV and images
    uploaded_csv = st.file_uploader("Upload CSV data", type=['csv'])
    uploaded_image = st.file_uploader("Upload leaf image", type=['jpg','png'])
else:
    # Initialize cloud mode variables as None
    uploaded_csv = None
    uploaded_image = None







# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .led-indicator {
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        margin: 10px;
        border: 2px solid #333;
    }
    .led-red-on {
        background-color: #ff0000;
        box-shadow: 0 0 15px #ff0000;
    }
    .led-red-off {
        background-color: #660000;
    }
    .led-green-on {
        background-color: #00ff00;
        box-shadow: 0 0 15px #00ff00;
    }
    .led-green-off {
        background-color: #006600;
    }
    .status-healthy {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .status-unhealthy {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
    }
    .calibration-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #dee2e6;
    }
    .data-section {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .file-info {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'measurement_history' not in st.session_state:
    st.session_state.measurement_history = []

# Hardware data paths (matching your hardware code)
desktop_path = os.path.expanduser("~/Desktop")
base_folder = os.path.join(desktop_path, "BTech Project")
healthy_folder = os.path.join(base_folder, "Healthy Leaf")
unhealthy_folder = os.path.join(base_folder, "Unhealthy Leaf")
sample_images_folder = os.path.join(desktop_path, "SampleImages")

# Configuration
BAND_LABELS = [f"Band{i+1}" for i in range(18)]
SPECTRAL_BANDS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'R', 'S', 'T', 'U', 'V', 'W']
DEFAULT_THRESHOLDS = {
    'chlorophyll': 2.5,
    'vitamin_a': 1.8,
    'vitamin_c': 2.0,
    'moisture': 2.0
}

def get_latest_csv_data():
    """Get the latest data from the CSV files created by hardware"""
    try:
        # Check both healthy and unhealthy folders for CSV files
        csv_files = []
        
        for folder in [healthy_folder, unhealthy_folder]:
            if os.path.exists(folder):
                csv_files.extend(glob.glob(os.path.join(folder, "*.csv")))
        
        if not csv_files:
            return None, None, "No CSV files found in project folders"
        
        # Get the most recent CSV file
        latest_csv = max(csv_files, key=os.path.getctime)
        
        # Read the CSV file
        df = pd.read_csv(latest_csv)
        
        if len(df) == 0:
            return None, None, "CSV file is empty"
        
        # Get the latest row (most recent measurement)
        latest_row = df.iloc[-1]
        
        # Extract the 18 spectral bands
        sensor_data = []
        for i in range(18):
            band_col = f"Band{i+1}"
            if band_col in df.columns:
                sensor_data.append(float(latest_row[band_col]))
            else:
                sensor_data.append(0.1)  # Default value if column missing
        
        # Determine health status from the CSV file itself
        health_status = None
        status_columns = ['Chlorophyll Status', 'Vitamin A Status', 'Vitamin C Status', 'Moisture Status']
        if all(col in df.columns for col in status_columns):
            statuses = [latest_row[col] for col in status_columns]
            health_status = {
                'chlorophyll': latest_row['Chlorophyll Status'],
                'vitamin_a': latest_row['Vitamin A Status'],
                'vitamin_c': latest_row['Vitamin C Status'],
                'moisture': latest_row['Moisture Status'],
                'is_healthy': all(status == 'Sufficient' for status in statuses)
            }
        
        # Get timestamp
        timestamp = latest_row['Timestamp'] if 'Timestamp' in df.columns else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        return sensor_data, health_status, f"Data from: {os.path.basename(latest_csv)} ({timestamp})"
    
    except Exception as e:
        return None, None, f"Error reading CSV files: {str(e)}"

def get_latest_image():
    """Get the latest image from SampleImages folder"""
    try:
        if not os.path.exists(sample_images_folder):
            return None, None, "SampleImages folder not found"
        
        # Look for image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(sample_images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(sample_images_folder, ext.upper())))
        
        if not image_files:
            return None, None, "No images found in SampleImages folder"
        
        # Get the most recent image
        latest_image_file = max(image_files, key=os.path.getctime)
        
        # Get file creation time
        file_time = os.path.getctime(latest_image_file)
        timestamp = datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
        
        # Load the image
        image = Image.open(latest_image_file)
        
        return image, timestamp, f"Latest image: {os.path.basename(latest_image_file)}"
        
    except Exception as e:
        return None, None, f"Error loading image: {str(e)}"

def get_latest_graphs():
    """Get the latest generated graphs from the hardware"""
    try:
        graph_files = []
        graph_info = []
        
        for folder in [healthy_folder, unhealthy_folder]:
            if os.path.exists(folder):
                png_files = glob.glob(os.path.join(folder, "*.png"))
                for png_file in png_files:
                    file_time = os.path.getctime(png_file)
                    graph_files.append((png_file, file_time))
        
        if not graph_files:
            return [], "No graph files found"
        
        # Sort by creation time and get the 4 most recent graphs
        graph_files.sort(key=lambda x: x[1], reverse=True)
        recent_graphs = graph_files[:4]
        
        for graph_path, _ in recent_graphs:
            filename = os.path.basename(graph_path)
            # Parse the graph type from filename
            if "Chlorophyll" in filename:
                graph_type = "Chlorophyll"
            elif "Vitamin A" in filename:
                graph_type = "Vitamin A"
            elif "Vitamin C" in filename:
                graph_type = "Vitamin C"
            elif "Moisture" in filename:
                graph_type = "Moisture"
            else:
                graph_type = "Unknown"
            
            graph_info.append({
                'path': graph_path,
                'type': graph_type,
                'filename': filename
            })
        
        return graph_info, f"Found {len(graph_info)} recent graphs"
    
    except Exception as e:
        return [], f"Error loading graphs: {str(e)}"

def analyze_leaf_health_from_data(sensor_data, thresholds):
    """Analyze leaf health based on spectral bands (matching hardware logic)"""
    if not sensor_data or len(sensor_data) < 18:
        return None
    
    # Calculate signals using the same logic as hardware
    chl_signal = sum(sensor_data[7:13])       # H to L (bands 8-13)
    vit_a_signal = sum(sensor_data[13:18])    # R to W (bands 14-18)
    vit_c_signal = sum(sensor_data[4:7])      # E, F, G (bands 5-7)
    moisture_signal = sum(sensor_data[1:4])   # B, C, D (bands 2-4)
    
    # Determine status
    chl_status = "Sufficient" if chl_signal > thresholds['chlorophyll'] else "Low"
    vit_a_status = "Sufficient" if vit_a_signal > thresholds['vitamin_a'] else "Low"
    vit_c_status = "Sufficient" if vit_c_signal > thresholds['vitamin_c'] else "Low"
    moisture_status = "Sufficient" if moisture_signal > thresholds['moisture'] else "Low"
    
    is_healthy = all(status == "Sufficient" for status in [chl_status, vit_a_status, vit_c_status, moisture_status])
    
    return {
        'is_healthy': is_healthy,
        'chlorophyll': {'status': chl_status, 'signal': chl_signal, 'bands': sensor_data[7:13]},
        'vitamin_a': {'status': vit_a_status, 'signal': vit_a_signal, 'bands': sensor_data[13:18]},
        'vitamin_c': {'status': vit_c_status, 'signal': vit_c_signal, 'bands': sensor_data[4:7]},
        'moisture': {'status': moisture_status, 'signal': moisture_signal, 'bands': sensor_data[1:4]}
    }

def create_gauge_chart(value, title, min_val=0, max_val=5, threshold=2.5):
    """Create analog-style gauge meter"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': threshold},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold], 'color': 'lightgray'},
                {'range': [threshold, max_val], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_spectral_plot(bands, title, color, band_names):
    """Create spectral analysis plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=band_names,
        y=bands,
        mode='lines+markers',
        name=title,
        line=dict(color=color, width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"{title} Spectral Analysis",
        xaxis_title="Spectral Band",
        yaxis_title="Normalized Value",
        template="plotly_white",
        height=300
    )
    
    return fig

# Main Interface
st.markdown('<h1 class="main-header">🌿 Leaf Health Monitoring System</h1>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Hardware Status
    st.subheader("🔧 Hardware Integration")
    st.text("📁 Project Folder:")
    st.code(base_folder)
    st.text("📷 Images Folder:")
    st.code(sample_images_folder)
    
    # Check if folders exist
    folders_exist = all(os.path.exists(folder) for folder in [base_folder, sample_images_folder])
    if folders_exist:
        st.success("✅ Hardware folders detected")
    else:
        st.error("❌ Hardware folders not found")
        st.info("Run the hardware code first to create the folder structure")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 2, 30, 5)
    
    st.divider()
    
    # File Status
    st.subheader("📊 Data Status")
    
    # Check for latest files
    sensor_data, health_status, sensor_msg = get_latest_csv_data()
    image, img_timestamp, img_msg = get_latest_image()
    graphs, graph_msg = get_latest_graphs()
    
    st.markdown(f'<div class="file-info">📈 Data: {sensor_msg}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="file-info">📷 Image: {img_msg}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="file-info">📊 Graphs: {graph_msg}</div>', unsafe_allow_html=True)
    
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    st.divider()
    
    # Threshold Configuration
    st.subheader("🎯 Analysis Thresholds")
    st.info("These should match your hardware thresholds")
    thresholds = {}
    thresholds['chlorophyll'] = st.slider("Chlorophyll Threshold", 1.0, 5.0, DEFAULT_THRESHOLDS['chlorophyll'], 0.1)
    thresholds['vitamin_a'] = st.slider("Vitamin A Threshold", 1.0, 4.0, DEFAULT_THRESHOLDS['vitamin_a'], 0.1)
    thresholds['vitamin_c'] = st.slider("Vitamin C Threshold", 1.0, 4.0, DEFAULT_THRESHOLDS['vitamin_c'], 0.1)
    thresholds['moisture'] = st.slider("Moisture Threshold", 1.0, 4.0, DEFAULT_THRESHOLDS['moisture'], 0.1)

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    # Hardware Status Section
    st.markdown('<div class="calibration-section">', unsafe_allow_html=True)
    st.header("🔧 Hardware Integration Status")
    
    # Check hardware connection status
    if folders_exist and sensor_data:
        st.success("✅ **HARDWARE CONNECTED** - Data is being received!")
        
        # LED Status indicators (based on latest data)
        col_led1, col_led2, col_led3 = st.columns([1, 1, 2])
        
        with col_led1:
            st.markdown("**Red LED Status**")
            if health_status and not health_status['is_healthy']:
                st.markdown('<div class="led-indicator led-red-on"></div>', unsafe_allow_html=True)
                st.caption("🔴 Unhealthy leaf detected")
            else:
                st.markdown('<div class="led-indicator led-red-off"></div>', unsafe_allow_html=True)
                st.caption("⚪ Off")
        
        with col_led2:
            st.markdown("**Green LED Status**")
            if health_status and health_status['is_healthy']:
                st.markdown('<div class="led-indicator led-green-on"></div>', unsafe_allow_html=True)
                st.caption("🟢 Healthy leaf detected")
            else:
                st.markdown('<div class="led-indicator led-green-off"></div>', unsafe_allow_html=True)
                st.caption("⚪ Off")
        
        with col_led3:
            st.markdown("**System Status**")
            st.success("✅ Ready and monitoring")
            st.caption("Hardware is running and collecting data")
        
    else:
        st.warning("⚠️ **HARDWARE NOT DETECTED**")
        st.info("Make sure the hardware code is running and has collected at least one measurement.")
        
        # Show offline LED indicators
        col_led1, col_led2, col_led3 = st.columns([1, 1, 2])
        
        with col_led1:
            st.markdown("**Red LED**")
            st.markdown('<div class="led-indicator led-red-off"></div>', unsafe_allow_html=True)
        
        with col_led2:
            st.markdown("**Green LED**")
            st.markdown('<div class="led-indicator led-green-off"></div>', unsafe_allow_html=True)
        
        with col_led3:
            st.markdown("**Status:** Offline")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Display Section
    if sensor_data:
        st.markdown('<div class="data-section">', unsafe_allow_html=True)
        st.header("🔬 Live Data from Hardware")
        
        # Show raw spectral data
        st.subheader("🌈 18-Band Spectral Data")
        
        # Create full spectrum plot
        fig_full = go.Figure()
        fig_full.add_trace(go.Scatter(
            x=SPECTRAL_BANDS,
            y=sensor_data,
            mode='lines+markers',
            name="Live Spectrum",
            line=dict(color="#8A2BE2", width=3),
            marker=dict(size=8)
        ))
        
        fig_full.update_layout(
            title="Real-time 18-Band Spectral Analysis",
            xaxis_title="Spectral Band",
            yaxis_title="Sensor Value",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_full, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.header("📊 Live Status")
    
    if not sensor_data:
        st.info("⚠️ Waiting for hardware data...")
        st.markdown("Run the hardware script and perform a leaf scan to see live results here.")
    else:
        # Use health status from CSV if available, otherwise calculate
        if health_status:
            analysis = health_status
        else:
            analysis = analyze_leaf_health_from_data(sensor_data, thresholds)
        
        if analysis:
            # Overall Status
            if analysis['is_healthy']:
                st.markdown(
                    '<div class="status-healthy"><h3>🟢 HEALTHY LEAF</h3><p>All parameters sufficient</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="status-unhealthy"><h3>🔴 UNHEALTHY LEAF</h3><p>One or more parameters low</p></div>',
                    unsafe_allow_html=True
                )
            
            st.subheader("📈 Parameter Status")
            
            # Show individual statuses
            st.metric("🟢 Chlorophyll", 
                     analysis.get('chlorophyll', {}).get('status', 'Unknown'),
                     f"Signal: {analysis.get('chlorophyll', {}).get('signal', 0):.2f}" if 'chlorophyll' in analysis else "")
            
            st.metric("🟡 Vitamin A", 
                     analysis.get('vitamin_a', {}).get('status', 'Unknown'),
                     f"Signal: {analysis.get('vitamin_a', {}).get('signal', 0):.2f}" if 'vitamin_a' in analysis else "")
            
            st.metric("🟠 Vitamin C", 
                     analysis.get('vitamin_c', {}).get('status', 'Unknown'),
                     f"Signal: {analysis.get('vitamin_c', {}).get('signal', 0):.2f}" if 'vitamin_c' in analysis else "")
            
            st.metric("💧 Moisture", 
                     analysis.get('moisture', {}).get('status', 'Unknown'),
                     f"Signal: {analysis.get('moisture', {}).get('signal', 0):.2f}" if 'moisture' in analysis else "")

# Display Latest Image
if image:
    st.header("📷 Latest Captured Image")
    col_img1, col_img2 = st.columns([2, 1])
    
    with col_img1:
        st.image(image, 
                caption=f"Captured: {img_timestamp}", 
                use_column_width=True)
    
    with col_img2:
        st.subheader("Image Info")
        st.text(f"Timestamp: {img_timestamp}")
        st.text(f"Size: {image.size}")
        st.text(f"Mode: {image.mode}")

# Display Hardware-Generated Graphs
if graphs:
    st.header("📊 Hardware-Generated Analysis Graphs")
    
    # Organize graphs by type
    graph_dict = {}
    for graph in graphs:
        graph_dict[graph['type']] = graph
    
    # Display graphs in a 2x2 grid
    if len(graph_dict) >= 2:
        col_g1, col_g2 = st.columns(2)
        
        graph_types = list(graph_dict.keys())
        
        with col_g1:
            for i, graph_type in enumerate(graph_types[::2]):  # Even indices
                if graph_type in graph_dict:
                    st.subheader(f"{graph_type} Analysis")
                    graph_image = Image.open(graph_dict[graph_type]['path'])
                    st.image(graph_image, caption=f"Hardware-generated {graph_type} graph", use_column_width=True)
        
        with col_g2:
            for i, graph_type in enumerate(graph_types[1::2]):  # Odd indices
                if graph_type in graph_dict:
                    st.subheader(f"{graph_type} Analysis")
                    graph_image = Image.open(graph_dict[graph_type]['path'])
                    st.image(graph_image, caption=f"Hardware-generated {graph_type} graph", use_column_width=True)
    
    else:
        # Display available graphs in single column
        for graph in graphs:
            st.subheader(f"{graph['type']} Analysis")
            graph_image = Image.open(graph['path'])
            st.image(graph_image, caption=f"Hardware-generated {graph['type']} graph", use_column_width=True)

# CSV Data Viewer
if sensor_data:
    st.header("📋 Raw CSV Data from Hardware")
    
    try:
        # Load and display CSV data
        csv_files = []
        
        for folder in [healthy_folder, unhealthy_folder]:
            if os.path.exists(folder):
                csv_files.extend(glob.glob(os.path.join(folder, "*.csv")))
        
        if csv_files:
            # Get the most recent CSV
            latest_csv = max(csv_files, key=os.path.getctime)
            df = pd.read_csv(latest_csv)
            
            # Show the last few rows
            st.subheader(f"Latest Data from: {os.path.basename(latest_csv)}")
            st.dataframe(df.tail(10), use_container_width=True)
            
            # Download option
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full CSV Data",
                data=csv_data,
                file_name=f"hardware_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error loading CSV data: {str(e)}")

# System Information
st.header("ℹ️ System Information")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.subheader("🔧 Hardware Status")
    st.text(f"Base Folder: {'✅ Found' if os.path.exists(base_folder) else '❌ Missing'}")
    st.text(f"Healthy Folder: {'✅ Found' if os.path.exists(healthy_folder) else '❌ Missing'}")
    st.text(f"Unhealthy Folder: {'✅ Found' if os.path.exists(unhealthy_folder) else '❌ Missing'}")
    st.text(f"Images Folder: {'✅ Found' if os.path.exists(sample_images_folder) else '❌ Missing'}")

with col_info2:
    st.subheader("📊 Data Status")
    if sensor_data:
        st.text(f"Live Data: ✅ Available")
        st.text(f"Spectral Bands: {len(sensor_data)}")
        st.text(f"Latest Image: {'✅ Available' if image else '❌ None'}")
        st.text(f"Graphs: {len(graphs)} available")
    else:
        st.text("Live Data: ❌ No data")
        st.text("Run hardware script first")

with col_info3:
    st.subheader("⚙️ Thresholds")
    st.text(f"Chlorophyll: {thresholds['chlorophyll']}")
    st.text(f"Vitamin A: {thresholds['vitamin_a']}")
    st.text(f"Vitamin C: {thresholds['vitamin_c']}")
    st.text(f"Moisture: {thresholds['moisture']}")

# Instructions
with st.expander("📋 Hardware Integration Instructions"):
    st.markdown("""
    ### How to Use This Dashboard with Your Hardware
    
    1. **Start the Hardware Script**: Run your Python hardware script (`paste.txt`) on your Raspberry Pi
    2. **Perform Calibration**: Follow the prompts in your hardware script to calibrate dark and white references
    3. **Scan Leaves**: Press Enter in your hardware script to scan leaves
    4. **View Results**: This dashboard will automatically display:
       - Latest spectral data from CSV files
       - Captured images from the camera
       - Health analysis graphs generated by hardware
       - Real-time LED status indicators
    
    ### File Structure Created by Hardware
    ```
    ~/Desktop/BTech Project/
    ├── Healthy Leaf/
    │   ├── leaf_analysis_healthy.csv
    │   └── [Generated graphs]
    ├── Unhealthy Leaf/
    │   ├── leaf_analysis_unhealthy.csv
    │   └── [Generated graphs]
    └── SampleImages/
        └── [Captured images]
    ```
    
    
    
    ### Features
    - 🔄 **Auto-refresh**: Dashboard updates automatically as hardware collects data
    - 📊 **Live Monitoring**: Real-time display of sensor readings and health status
    - 🚦 **LED Indicators**: Visual representation of hardware LED status
    - 📈 **Hardware Graphs**: Display graphs generated by your hardware script
    - 📷 **Image Integration**: Shows latest captured leaf images
    
    ### Troubleshooting
    - **No data showing**: Make sure hardware script is running and has completed at least one scan
    - **Folders not found**: Run hardware script first to create folder structure
    - **Old data**: Check auto-refresh settings or click refresh button
    """)

# Footer
st.markdown("---")
st.markdown("🌿 **Leaf Health Monitoring System** | Hardware Integration Mode | Built with Streamlit")
st.markdown("🔗 **Connected to:** Raspberry Pi Hardware | 📊 **Live Data:** CSV Files | 📷 **Images:** libcamera-still")