import streamlit as st

# Configure the main window settings
st.set_page_config(
    page_title="Project Alpha | Interactive Sandbox",
    page_icon="🔬",
    layout="wide"
)

st.title("Welcome to the Project Alpha Sandbox 🔬")

st.markdown("""
This interactive platform is designed to help you visually explore the **Project Alpha** dataset alongside our weekly Jupyter Notebook tasks.

## How to use this sandbox:

### **Select a weekly module from the sidebar**  for that specific topic.


#### *Note: If the sidebar is hidden, click the small arrow icon in the top left corner of your screen to reveal the navigation menu.*
""")