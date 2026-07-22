import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Water Bottle Detection AI",
    page_icon="🍼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# LOAD CSS
# ==========================================
try:
    with open("assets/style.css", "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except FileNotFoundError:
    st.warning("style.css file not found.")


# ==========================================
# EXTRA CSS
# ==========================================
st.markdown("""
<style>

/* Main Titles */
h1, h1 span{
    color:#00E5FF !important;
    font-weight:700 !important;
    text-shadow:0 0 10px rgba(0,229,255,.25);
}

/* Sidebar */
[data-testid="stSidebar"] h1{
    color:#00E5FF !important;
}

/* Headings */
h2,h3,h4,h5,h6{
    color:#E0FFFF !important;
}

/* Text */
html,body,p,span,li,label{
    color:white !important;
}

/* Sidebar Text */
[data-testid="stSidebar"] *,
[data-testid="stSidebarNav"] *{
    color:white !important;
}

/* Metrics */
[data-testid="stMetricLabel"]{
    color:white !important;
}

[data-testid="stMetricValue"]{
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.image(
    "https://img.icons8.com/color/96/water-bottle.png",
    width=80
)

st.sidebar.title("🍼 Water Bottle AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📂 Select Module",
    (
        "🏠 Home",
        "🖼 Image Detection",
        "🎥 Video Detection",
        "📷 Webcam Detection",
        "📊 Dashboard",
        "ℹ About",
    ),
)

st.sidebar.markdown("---")

st.sidebar.success("✅ YOLO11 Model Loaded")

# ==========================================
# HOME
# ==========================================

if page == "🏠 Home":

    st.title("🍼 Water Bottle Detection AI")

    st.subheader(
        "Real-Time Bottle Detection using YOLO11 • Streamlit • OpenCV"
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🖼 Images", "Ready")
    c2.metric("🎥 Videos", "Ready")
    c3.metric("📷 Webcam", "Ready")
    c4.metric("🤖 Model", "YOLO11")

    st.markdown("---")

    st.subheader("🚀 Features")

    left, right = st.columns(2)

    with left:
        st.success("✅ Image Detection")
        st.success("✅ Video Detection")
        st.success("✅ Live Webcam Detection")

    with right:
        st.success("✅ Analytics Dashboard")
        st.success("✅ Download Results")
        st.success("✅ YOLO11 AI Model")

    st.markdown("---")

    st.info(
        "👈 Select any module from the sidebar to start detection."
    )

# ==========================================
# IMAGE
# ==========================================

elif page == "🖼 Image Detection":

    exec(open("modules/image.py", encoding="utf-8").read())

# ==========================================
# VIDEO
# ==========================================

elif page == "🎥 Video Detection":

    exec(open("modules/video.py", encoding="utf-8").read())

# ==========================================
# WEBCAM
# ==========================================

elif page == "📷 Webcam Detection":

    exec(open("modules/webcam.py", encoding="utf-8").read())

# ==========================================
# DASHBOARD
# ==========================================

elif page == "📊 Dashboard":

    exec(open("modules/dashboard.py", encoding="utf-8").read())

# ==========================================
# ABOUT
# ==========================================

elif page == "ℹ About":

    exec(open("modules/about.py", encoding="utf-8").read())

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "© 2026 Water Bottle Detection AI | Powered by YOLO11 • Streamlit • OpenCV"
)