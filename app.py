import streamlit as st
import pandas as pd
import plotly.express as px

from parser import parse
from ramanTools import bubblefill, cosmic_rays_removal, SNV, savgol_filter


if __name__ == '__main__':
    st.set_page_config(page_title="Raman Spectroscopy")

    # Sidebar
    st.sidebar.header("Renishaw Acquisition")
    files = st.sidebar.file_uploader("Spectral data", ["wdf"], True)
    df = pd.DataFrame({file.name: parse(file) for file in files})
    st.sidebar.metric("Total Acquisitions", f"{len(df.columns)} spectra")
    start, stop = st.sidebar.select_slider(
            "Wavelength range",
            options=df.index,
            value=(df.index.min(), df.index.max()))
    st.sidebar.header("Preprocessing")
    preprocessing = st.sidebar.multiselect(
            "Preprocessing",
            ["Cosmic Ray Removal", "Savgol", "Raman", "SNV"],
            "Raman")
    with st.sidebar.expander("Parameters"):
        window_length = int(st.number_input("Window length", 0, value=11))
        polyorder = int(st.number_input("Polyorder", 0, value=3))
        bubblewidths = int(st.number_input("Bubble widths", 0, value=40))
    if "Cosmic Ray Removal" in preprocessing:
        df = df.apply(cosmic_rays_removal, raw=True)
    if "Savgol" in preprocessing:
        df = df.apply(savgol_filter, raw=True, window_length=window_length, polyorder=polyorder)
    if "Raman" in preprocessing:
        df = df.apply(bubblefill, raw=True, bubblewidths=bubblewidths)
    if "SNV" in preprocessing:
        df = df.apply(SNV, raw=True)

    # Main
    with open("README.md", "r") as readme:
        st.title(readme.readline().strip('#').strip())
        st.markdown(readme.read())
    df = df.loc[start:stop]
    fig = px.line(df, labels={'value': 'Absorption', 'wavelength': 'Wavelength (nm)'})
    fig.update_layout(
            showlegend=False,
            title={
                    'text': "Raw Acquisition",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)
