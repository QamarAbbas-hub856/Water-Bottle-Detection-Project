import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ==========================
# Text Color & Style Customization (CSS)
# ==========================
st.markdown(
    """
    <style>
    /* Sab text aur main titles ko white karne ke liye */
    html, body, [data-testid="stWidgetLabel"], h1, h2, h3, h4, h5, h6, p, span {
        color: #ffffff !important;
    }
    
    /* Sidebar ke title ("Water Bottle AI"), menus aur links ko white karne ke liye */
    [data-testid="stSidebar"] *, [data-testid="stSidebarNav"] *, [data-testid="stSidebarUserContent"] * {
        color: #ffffff !important;
    }
    
    /* Sidebar ke radio options aur markdown labels ko specific target karne ke liye */
    div[data-testid="stRadio"] label span {
        color: #ffffff !important;
    }
    
    [data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
    }

    /* Metrics labels (Total Detections, etc.) ko clear white karne ke liye */
    [data-testid="stMetricLabel"] p {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    /* Metrics values (numbers) ko bhi clear white karne ke liye */
    [data-testid="stMetricValue"] > div {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Detection Analytics Dashboard")

FILE = "data/history.csv"

# ==========================
# Check File
# ==========================

if not os.path.exists(FILE):
    st.warning("⚠ No detection history found.")
    st.stop()

df = pd.read_csv(FILE)

if df.empty:
    st.info("No detection data available.")
    st.stop()

# ==========================
# Add Missing Columns
# ==========================

if "Mode" not in df.columns:
    df["Mode"] = "Image"

if "Bottle_Count" not in df.columns:
    df["Bottle_Count"] = 0

if "Average_Confidence" not in df.columns:

    if "Confidence" in df.columns:
        df["Average_Confidence"] = df["Confidence"]
    else:
        df["Average_Confidence"] = 0

# ==========================
# Sidebar Filter
# ==========================

mode = st.sidebar.selectbox(
    "Detection Mode",
    ["All", "Image", "Video", "Webcam"]
)

if mode != "All":
    df = df[df["Mode"] == mode]

# ==========================
# Statistics
# ==========================

total_detection = len(df)

total_bottles = int(df["Bottle_Count"].sum())

avg_conf = round(df["Average_Confidence"].mean(), 2)

c1, c2, c3 = st.columns(3)

c1.metric("📷 Total Detections", total_detection)

c2.metric("🍼 Total Bottles", total_bottles)

c3.metric("🎯 Avg Confidence", f"{avg_conf}%")

st.markdown("---")

# ==========================
# Charts
# ==========================

left, right = st.columns(2)

with left:

    st.subheader("🍼 Bottle Count")

    st.bar_chart(df["Bottle_Count"])

with right:

    st.subheader("🎯 Confidence")

    st.line_chart(df["Average_Confidence"])

st.markdown("---")

# ==========================
# Pie Chart (Size Controlled)
# ==========================

st.subheader("📊 Detection Mode Distribution")

col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='none')
    ax.set_facecolor('none')
    
    df["Mode"].value_counts().plot.pie(
        autopct="%1.1f%%",
        ax=ax,
        textprops={'color': 'white'}
    )

    ax.set_ylabel("")
    
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ==========================
# Detection History
# ==========================

st.subheader("📋 Detection History")

st.dataframe(
    df,
    width="stretch",
    hide_index=True
)

st.markdown("---")

# ==========================
# Download CSV
# ==========================

with open(FILE, "rb") as f:

    st.download_button(
        "⬇ Download History CSV",
        data=f,
        file_name="history.csv",
        mime="text/csv",
        width="stretch"
    )