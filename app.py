import streamlit as st
import os
from main import speed_detector

# Create a Streamlit app to upload a video and display the output video after running the speed detection algorithm

# Page title and description
st.title("Speed Detection App")
st.write("Upload a video file to detect vehicle speeds.")

# File uploader for video input
uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = "input_video.mp4"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Run speed detection on the uploaded video
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    detector = speed_detector(args=None, SOURCE_VIDEO_PATH=file_path, output_dir=output_dir)
    # detector.ready()

    # Display the output video
    output_video_path = os.path.join(output_dir, "vehicles-result.mp4")
    # Convert the video to a format that Streamlit can display
    converted_video_path = "output_video_streamlit.mp4"
    os.system(f"ffmpeg -i {output_video_path} -c:v libx264 -pix_fmt yuv420p {converted_video_path}")
    st.write("Output video:")
    st.video(open(converted_video_path, 'rb').read())
