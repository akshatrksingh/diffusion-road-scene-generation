import streamlit as st

# Basic page setup first
st.set_page_config(page_title="Road Scene Generator", page_icon="🚗", layout="centered")

# Initial loading message
st.title("AI-Powered Road Scene Generator")
st.write("Generate road scene images using text descriptions and AI-based diffusion models.")

# Loading message for model initialization
with st.spinner("There's some magic going on behind the scenes... This may take a few minutes..."):
    from generator import RoadSceneGenerator
    generator = RoadSceneGenerator()

# Once loaded, show the main interface
st.write("Models loaded! Ready to generate scenes.")

# User Input
description = st.text_input("Enter a road scene description:", "A highway at night")

# Button to trigger image generation
if st.button("Generate Image"):
    with st.spinner("Generating your road scene... This may take a few minutes."):
        image = generator.generate_scene(description)
    
    # Display the generated image
    st.image(image, caption="Generated Scene", use_column_width=True)
    
    # Success message
    st.success("Scene generation complete!")