import streamlit as st
import pandas as pd

# Title and description
st.title("My Project Title")
st.write("A short description of what this app does.")

# Load data
df = pd.read_csv("csv(s)/exercise.csv")  # use your actual file name

# Show the data
st.subheader("Dataset")
st.dataframe(df)

# Show a chart
st.subheader("Visualization")
st.line_chart(df)
