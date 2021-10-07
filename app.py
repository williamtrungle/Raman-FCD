import streamlit as st
import pandas as pd
import plotly.express as px

from parser import parse


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
