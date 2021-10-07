import uuid
import streamlit as st
import pandas as pd
import plotly.express as px

from parser import readme, parse, preprocess, STEPS

LABELS = {
    'value': 'Absorption',
    'wavelength': 'Wavelength (nm)',
    'variable': 'Acquisition',
}

def plot(df, placeholder=None, showlegend=False, labels=None, hovermode='closest', title=''):
    fig = px.line(df, labels=labels)
    fig.update_layout(
            showlegend=showlegend,
            hovermode=hovermode,
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

    # Description
    title, body = readme()
    st.title(title)
    st.markdown(body+"To begin, upload data using the sidebar.")
    plot(df, labels=LABELS, title="Raw Acquisitions")
    st.caption("Figure 1. Raw acquisition data after preprocessing. Steps include: "+', '.join(steps)+'.')
