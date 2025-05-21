# Streamlit App

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import add_instruments

# Sample Streamlit app displaying a matplotlib plot
def main():
    st.title("Sample Yield Curve Plot in Streamlit")

    # Display the plot in Streamlit
    fig = add_instruments.main()   # actually call the function
    st.pyplot(fig)  
if __name__ == "__main__":
    main()
