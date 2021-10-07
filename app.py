import streamlit as st


with open("README.md", "r") as readme:
    title = readme.readline().strip('#').strip()
    description = readme.read()


if __name__ == '__main__':
    st.set_page_config(page_title="FCD")
    st.title(title)
    st.markdown(description)
