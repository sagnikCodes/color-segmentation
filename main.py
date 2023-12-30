import streamlit as st
import matplotlib.pyplot as plt
import base64
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# Custom CSS styles
custom_css = """
    <style>
        .appview-container {
            background-color: #F1EAFF;
        }
        .st-emotion-cache-18ni7ap {
            background-color: #DCBFFF;
        }
        .st-emotion-cache-1gulkj5 {
            background-color: #E5D4FF;
        }
        .st-emotion-cache-1on073z {
            background-color: #D0A2F7;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stMarkdown h1 {
            text-align: center;
            color: #7743DB;  
        }
    </style>
"""

st.title("Color Segmentation")

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image_str = base64.b64encode(image_bytes).decode()

    st.markdown(
        f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{image_str}" alt="Uploaded Image" style="height:300px;"></div>',
        unsafe_allow_html=True,
    )

    image = plt.imread(uploaded_file)
    image = image.reshape(-1, 3)

    user_number = int(st.number_input("Enter how many dominant colors you want to see", min_value=0, max_value=100, value=5, step=1))
    kmeans = KMeans(n_clusters=user_number)
    kmeans.fit(image)

    # Plotly Bar Chart with Hover Text
    colors_hex = ['#%02x%02x%02x' % tuple(rgb) for rgb in kmeans.cluster_centers_.astype(int)]

    fig = go.Figure()
    st.write(f"Following are the top-{user_number} dominant colors in the image you uploaded")
    for i, color_hex in enumerate(colors_hex):
        fig.add_trace(go.Bar(
            x=[i],
            y=[1],
            marker=dict(color=color_hex),
            hoverinfo="text",
            hovertemplate=f"rgb({', '.join(map(str, kmeans.cluster_centers_[i].astype(int)))})",
            showlegend=False
        ))

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=125,
        margin=dict(t=0, b=0, l=0, r=0),
        bargap=0  # Set the gap between bars to 0
    )

    # Hide plotly options
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
