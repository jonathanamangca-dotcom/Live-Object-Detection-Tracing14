import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import time
from pathlib import Path

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Live Object Detection & Tracing",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern background designs and styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e1e24 0%, #252830 100%);
        color: #ffffff;
    }
    .alert-box {
        background-color: rgba(220, 53, 69, 0.2);
        border: 1px solid #dc3545;
        color: #ff8585;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
    .success-box {
        background-color: rgba(40, 167, 69, 0.2);
        border: 1px solid #28a745;
        color: #75b798;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
    .warning-box {
        background-color: rgba(255, 193, 7, 0.2);
        border: 1px solid #ffc107;
        color: #ffc107;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Create a directory to save captured images
SAVE_DIR = Path("saved_frames")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize session state variables safely
if "detected_count" not in st.session_state:
    st.session_state.detected_count = 0
if "alert_triggered" not in st.session_state:
    st.session_state.alert_triggered = False
if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None
if "stream_active" not in st.session_state:
    st.session_state.stream_active = False

# Cache the model so it doesn't reload on every rerun
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Control Panel")
selected_class = st.sidebar.selectbox(
    "Select Target Object to Count & Track:",
    ("person", "cell phone", "laptop", "cup", "bottle", "chair", "book")
)

# Alert threshold configuration
threshold = st.sidebar.slider("Alert Threshold (Min. Objects Required)", 1, 10, 1)

# Action controls
save_image_btn = st.sidebar.button("Capture & Save Current Frame")

# --- Main Layout ---
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify, track, and monitor them in real-time.")

# Create columns for layout
col1, col2 = st.columns([2, 1])

# Use a custom video processor to manage state updates dynamically and safely across threads
class VisionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 tracking
        results = model.track(
            img,
            persist=True,
            conf=0.4,
            verbose=False
        )

        annotated_frame = results[0].plot()
        
        count = 0
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                if class_name == selected_class:
                    count += 1
        
        # Update session state
        st.session_state.detected_count = count
        st.session_state.alert_triggered = (count >= threshold)
        st.session_state.latest_frame = img.copy()
        st.session_state.stream_active = True

        # Use st.rerun to ensure elements reflect accurately inside Streamlit during execution
        try:
            st.rerun()
        except Exception:
            pass

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

with col1:
    st.subheader("Live Feed Stream")
    
    # Start WebRTC streamer
    webrtc_streamer(
        key="object-detection",
        video_processor_factory=VisionProcessor,
        async_processing=True,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 Analytics & Alerts")
    
    display_count = 0 if not st.session_state.stream_active else st.session_state.detected_count
    st.metric(label=f"Active Count: {selected_class}", value=display_count)
    
    # Display Alert Message based on condition
    if st.session_state.alert_triggered:
        st.markdown(
            f'<div class="alert-box">⚠️ ALERT: {selected_class} threshold of {threshold} reached!</div>', 
            unsafe_allow_html=True
        )
    
    # Save Image action with validation
    if save_image_btn:
        # Check if the WebRTC component is transmitting the video frame
        if st.session_state.stream_active and st.session_state.latest_frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = SAVE_DIR / f"frame_{timestamp}.jpg"
            cv2.imwrite(str(filename), st.session_state.latest_frame)
            
            st.markdown(
                f'<div class="success-box">✅ Frame saved to <br><code>{filename}</code></div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="warning-box">⚠️ Please turn on the camera using the WebRTC "Start" button first and wait for the feed to load!</div>',
                unsafe_allow_html=True
            )