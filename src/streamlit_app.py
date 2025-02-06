import streamlit as st
import os
os.environ["TORCH_ALLOW_CUSTOM_CLASS_LOADING"] = "1"

st.set_page_config(page_title="Road Scene Generator", page_icon="🚗", layout="centered")

# cache the generator initialization
@st.cache_resource
def load_generator():
    from generator import RoadSceneGenerator
    return RoadSceneGenerator()

st.title("AI-Powered Road Scene Generator")
st.write("Generate road scene images using text descriptions and AI-based diffusion models.")

try:
    with st.spinner("Loading models... This may take a few minutes..."):
        generator = load_generator()
    st.write("Models loaded! Ready to generate scenes.")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# User Input
description = st.text_input("Enter a road scene description:", "A highway at night")

if st.button("Generate Image"):
    try:
        with st.spinner("Generating your road scene... This may take a few minutes."):
            image = generator.generate_scene(description)
        
        st.image(image, caption="Generated Scene", use_column_width=True)
        st.success("Scene generation complete!")
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")