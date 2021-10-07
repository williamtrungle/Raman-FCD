import streamlit as st
import pandas as pd
import plotly.express as px

from parser import parse, preprocess, STEPS

LABELS = {
    'value': 'Absorption',
    'wavelength': 'Wavelength (nm)',
    'variable': 'Acquisition',
}

def plot(df, placeholder=None, showlegend=False, labels=None, title=''):
    fig = px.line(df, labels=labels)
    fig.update_layout(
            showlegend=showlegend,
            hovermode="x",
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
    (placeholder or st).plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    # Configuration
    st.set_page_config(page_title="Raman Spectroscopy")

    # Sidebar
    with st.sidebar.header("Renishaw Acquisition"):
        files = st.sidebar.file_uploader("Spectral data", ["wdf"], True)
        df = pd.DataFrame({file.name: parse(file) for file in files})
    with st.sidebar.header("Preprocessing"):
        start, stop = st.sidebar.select_slider(
                "Wavelength (nm)",
                options=df.index,
                value=(df.index.min(), df.index.max()))
        df = df.loc[start:stop]
        steps = st.sidebar.multiselect("Filters", STEPS, STEPS)
        with st.sidebar.expander("Parameters"):
            window_length = int(st.number_input("Window length", 0, value=11))
            polyorder = int(st.number_input("Polyorder", 0, value=3))
            bubblewidths = int(st.number_input("Bubble widths", 0, value=40))
        df = preprocess(
                df,
                *steps,
                window_length=window_length,
                polyorder=polyorder,
                bubblewidths=bubblewidths)

    # Main
    with open("README.md", "r") as readme:
        st.title(readme.readline().strip('#').strip())
        st.markdown(readme.read())
    with st.empty() as chart:
        plot(df, chart, labels=LABELS, title="Raw Acquisitions")
