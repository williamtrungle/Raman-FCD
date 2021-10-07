import streamlit as st

from memory_tempfile import MemoryTempfile


if __name__ == '__main__':
    st.set_page_config(page_title="Raman Spectroscopy")

    # Sidebar
    st.sidebar.header("Renishaw Acquisition")
    for file in st.sidebar.file_uploader("Spectral data", ["wdf"], True):
        with MemoryTempfile().TemporaryFile() as out:
            out.write(file.read())

    # Main
    with open("README.md", "r") as readme:
        st.title(readme.readline().strip('#').strip())
        st.markdown(readme.read())
