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
    st.set_page_config(page_title="Raman Spectroscopy", layout='wide')

    # Sidebar
    total = st.sidebar.empty()
    st.sidebar.header("Renishaw Acquisition")
    files = st.sidebar.file_uploader("Spectral data", ["wdf"], True)
    df = {}
    for file in files:
        try:
            data = parse(file)
        except ValueError:
            continue
        else:
            df[file.name[:-4]] = data
    df = pd.DataFrame(df)
    total.metric("Total", f"{len(df.columns)} files", f"{len(df.columns) - len(files)} errors")
    with st.sidebar.header("Preprocessing"):
        if not df.empty:
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
    _, col, _ = st.columns([1,2,1])
    title, body = readme()
    col.title(title)
    col.markdown(body+"To begin, upload data using the sidebar.")
    plot(df, labels=LABELS, title="Raw Acquisitions")
    _, col, _ = st.columns([1,2,1])
    col.caption(
            "Figure 1. "
            "Raw acquisition data after preprocessing. "
            "Steps include: "
            f"{', '.join(steps)}.")

    # Feature selection
    _, col, _ = st.columns([1,2,1])
    col.header("Feature selection")
    col.markdown("Create new features by averaging existing spectra together under a new name.")
    n = int(col.number_input("Number of features to create", 0))
    with col.form("Features"):
        features = {}
        for i in range(n):
            with st.expander(f"Feature {i}"):
                name = st.text_input("Name", key=f"Feature {i}")
                values = list(filter(lambda x: st.checkbox(x, key=f"Values {i}"), df.columns))
                features[name] = values
        st.form_submit_button("Combine")
    for name, col in features.items():
        df[name] = df[col].mean(axis=1)
    df = df.drop(columns=set(sum(features.values(), [])))
    plot(df, labels=LABELS, title="Combined Acquisitions", showlegend=True, hovermode='x')
    _, col, _ = st.columns([1,2,1])
    col.caption(
            "Figure 2. "
            "Feature extraction via averaging of selected spectra. "
            f"{'. '.join([f'{k}: {len(v)} spectra' for k, v in features.items()])}.")
