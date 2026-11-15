import streamlit as st


def display_image_comparison(left, right, label1, label2, width=900):
    """Swipe slider between two PIL images. Falls back to side-by-side if the widget is missing."""
    try:
        from streamlit_image_comparison import image_comparison
        image_comparison(img1=left, img2=right, label1=label1, label2=label2, width=width,
                         starting_position=50, show_labels=True, make_responsive=True, in_memory=True)
    except Exception:
        c1, c2 = st.columns(2)
        c1.image(left, caption=label1, use_container_width=True)
        c2.image(right, caption=label2, use_container_width=True)
