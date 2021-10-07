import streamlit as st
import pandas as pd

from parser import parse


if __name__ == '__main__':
    st.set_page_config(page_title="Raman Spectroscopy")

    # Sidebar
    st.sidebar.header("Renishaw Acquisition")
    df = pd.DataFrame({
        file.name: parse(file)
        for file in st.sidebar.file_uploader("Spectral data", ["wdf"], True)
    })
    st.sidebar.metric("Acquisitions", f"{len(df.columns)} spectra")

    # Main
    with open("README.md", "r") as readme:
        st.title(readme.readline().strip('#').strip())
        st.markdown(readme.read())
    st.line_chart(df)
