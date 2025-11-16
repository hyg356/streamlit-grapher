import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
st.title("3D Function Plotter")


st.write("Refer to the user manual in the 'home page', for any querries")

title = st.text_input("Enter title for plot: ")
x_name = st.text_input("Enter X-axis name: ")
y_name = st.text_input("Enter Y-axis name: ")
z_name = st.text_input("Enter Z-axis name: ")

num_funcs = st.number_input("Enter number of functions: ", min_value=1, step=1)

fig = go.Figure()

for i in range(num_funcs):
    st.text("NOTE- While using standard operations/ expressions, make sure you enter the prefix 'np.' before you enter the func/operation")
    expression = st.text_input(f"Enter func-{i+1} in terms of X and Y: ")

    X_domain_lower = st.number_input(f"Enter lower bound of 'X' in func-{i+1}: ", min_value=-500.0, max_value=500.0, step=0.001)
    X_domain_upper = st.number_input(f"Enter upper bound of 'X' in func-{i+1}: ", min_value=-500.0, max_value=500.0, step=0.001)

    Y_domain_lower = st.number_input(f"Enter lower bound of 'Y' in func-{i+1}: ", min_value=-500.0, max_value=500.0, step=0.001)
    Y_domain_upper = st.number_input(f"Enter upper bound of 'Y' in func-{i+1}: ", min_value=-500.0, max_value=500.0, step=0.001)
    

    X = np.linspace(X_domain_lower, X_domain_upper, 100)
    Y = np.linspace(Y_domain_lower, Y_domain_upper, 100)
    X, Y = np.meshgrid(X, Y)

    try:
        Z = eval(expression)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', name=f"Func {i+1}"))
    except Exception as e:
        st.error(f"Error in function {i+1}: {e}")

fig.update_layout(
    title=title,
    scene=dict(
        xaxis_title=x_name,
        yaxis_title=y_name,
        zaxis_title=z_name
    )
)

if st.button("Generate 3-D plot"):
    st.plotly_chart(fig, use_container_width=True)
