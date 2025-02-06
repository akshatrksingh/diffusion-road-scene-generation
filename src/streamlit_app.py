import streamlit as st
import os
import time
import threading
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable for PyTorch
os.environ["TORCH_ALLOW_CUSTOM_CLASS_LOADING"] = "1"

# Streamlit page configuration
st.set_page_config(page_title="Road Scene Generator", page_icon="🚗", layout="centered")

# Global flag to control the monitoring thread
stop_monitoring = False

def monitor_resources(cpu_limit=80, memory_limit=80):
    """
    Monitors CPU and memory usage in a separate thread.
    
    Args:
        cpu_limit (float): CPU usage limit (in percentage).
        memory_limit (float): Memory usage limit (in percentage).
    """
    global stop_monitoring
    while not stop_monitoring:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        logger.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_percent}%")
        
        if cpu_percent > cpu_limit:
            logger.warning(f"CPU usage exceeds limit: {cpu_percent}% > {cpu_limit}%")
            st.warning(f"CPU usage exceeds limit: {cpu_percent}% > {cpu_limit}%")
        
        if memory_percent > memory_limit:
            logger.warning(f"Memory usage exceeds limit: {memory_percent}% > {memory_limit}%")
            st.warning(f"Memory usage exceeds limit: {memory_percent}% > {memory_limit}%")
        
        time.sleep(1)

# Cache the generator initialization
@st.cache_resource
def load_generator():
    from generator import RoadSceneGenerator
    return RoadSceneGenerator()

# Streamlit app title and description
st.title("AI-Powered Road Scene Generator")
st.write("Generate road scene images using text descriptions and AI-based diffusion models.")

# Start the monitoring thread
if "monitor_thread" not in st.session_state:
    stop_monitoring = False
    monitor_thread = threading.Thread(target=monitor_resources, args=(80, 80))
    monitor_thread.daemon = True  # Daemonize thread to stop it when the main program exits
    monitor_thread.start()
    st.session_state.monitor_thread = monitor_thread

# User input for scene description
description = st.text_input("Enter a road scene description:", "A highway at night")

# Generate image on button click
if st.button("Generate Image"):
    try:
        with st.spinner("Generating your road scene... This may take a few minutes."):
            generator = load_generator()
            image = generator.generate_scene(description)
        
        st.image(image, caption="Generated Scene", use_column_width=True)
        st.success("Scene generation complete!")
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
    finally:
        # Stop the monitoring thread after image generation
        stop_monitoring = True
        st.session_state.monitor_thread.join()
        st.write("Resource monitoring stopped.")