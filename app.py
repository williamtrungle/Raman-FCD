import yaml
import uuid
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from parser import readme, parse, preprocess, STEPS, raman_shift_peak

LABELS = {
    'value': 'Absorption',
    'wavelength': 'Wavelength (nm)',
    'variable': 'Acquisition',
}

def plot(df,
        *lines,
        metadata=None,
        placeholder=None,
        showlegend=False,
        labels=None,
        hovermode='closest',
        title=''):
    fig = px.line(df, labels=labels)
    for line in lines:
        fig.add_vline(
                x=line[0],
                line_dash='dot',
                line_color='#ccc')
    for x, y in lines:
        fig.add_annotation(x=x, y=y, text=x, yshift=1)
    fig.update_traces(hovertemplate='%{y:.2f}')
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
    title, body = readme()

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
    col = st.columns([1,2,1])[1]
    col.title(title)
    col.markdown(body+"To begin, upload data using the sidebar.")
    plot(df, labels=LABELS, title="Raw Acquisitions")
    col = st.columns([1,2,1])[1]
    col.caption(
            "Figure 1. "
            "Raw acquisition data after preprocessing. "
            "Steps include: "
            f"{', '.join(steps)}.")

    # Feature selection
    col = st.columns([1,2,1])[1]
    col.header("Feature selection")
    col.markdown("Create new features by averaging existing spectra together under a new name.")
    n = int(col.number_input("Number of features to create", 0))
    with col.form("Features"):
        features = {}
        for i in range(n):
            with st.expander(f"Feature {i+1}"):
                name = st.text_input("Name", key=f"Feature {i+1}")
                values = list(filter(lambda x: st.checkbox(x, key=f"Values {i+1}"), df.columns))
                features[name] = values
        st.form_submit_button("Combine")
    for name, col in features.items():
        df[name] = df[col].mean(axis=1)
    df = df.drop(columns=set(sum(features.values(), [])))
    plot(df, labels=LABELS, title="Combined Acquisitions", showlegend=True, hovermode='x unified')
    col = st.columns([1,2,1])[1]
    col.caption(
            "Figure 2. "
            "Feature extraction via averaging of selected spectra. "
            f"{'. '.join([f'{k}: {len(v)} spectra' for k, v in features.items()])}.")

    # Band selection
    col = st.columns([1,2,1])[1]
    col.header("Band selection")
    col.markdown("Select band ranges in which peaks absorption occurs. "
                 "This allows identification of characteristic molecular "
                 "components of the scanned tissue "
                 "(*e.g.* characteristic proteins fingerprint).")
    n = int(col.number_input("Number of bands to create", 0))
    peaks = {}
    absorption = {}
    with col.form("Bands"):
        for i in range(n):
            start, stop = st.select_slider(
                    "Wavelength",
                    options=df.index,
                    value=(df.index.min(), df.index.max()),
                    key=f"Band {i+1}")
            p, a = zip(*[raman_shift_peak(df[col], start, stop) for col in df.columns])
            peaks[f"{start} - {stop} (nm)"] = pd.Series(p, index=df.columns)
            absorption[f"{start} - {stop} (nm)"] = pd.Series(a, index=df.columns)
        st.form_submit_button("Fingerprint")
    peaks = pd.DataFrame(peaks).T
    absorption = pd.DataFrame(absorption).T
    if not peaks.empty:
        col.dataframe(peaks)
        _, L, R, _ = st.columns([2,2,2,2])
        L.download_button("Download Fingerprint", peaks.to_csv(), "raman-fingerprint.csv")
        R.download_button("Download Combination", yaml.dump(features), "feature-selection.yaml")
        for col in df.columns:
            plot(df[col], *zip(peaks[col], absorption[col]), labels=LABELS, title=col, hovermode='x unified')
