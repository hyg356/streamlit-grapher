import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
st.title("2-D Plotter")


st.write("Refer to the user manual in the 'home page', for any querries")

title_chart = st.text_input("Enter title:")  
title_font_size = st.number_input("Enter font size of the title:", min_value=5, max_value=40, step=1)
x_name = st.text_input("Enter X-axis label:")
y_name = st.text_input("Enter Y-axis label:")
num_funcs = st.number_input("Enter the number of functions:", min_value=1, step=1)

fig = go.Figure()

for i in range(num_funcs):
    st.write("NOTE – use prefix 'np.' while entering built-in functions or constants (e.g., np.sin, np.pi)")
    expression = st.text_input(f"Enter func-{i+1} in terms of 'X':")
    domain_lower = st.number_input(f"Enter lower bound for func-{i+1}:", min_value=-500.0, max_value=500.0, step=0.001)
    domain_upper = st.number_input(f"Enter upper bound for func-{i+1}:", min_value=-500.0, max_value=500.0, step=0.001)

    if expression:
        try:
            # Plot main function curve
            X = np.linspace(domain_lower, domain_upper, 1000)
            Y = eval(expression)
            fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", name=f"func-{i+1}"))

            # Y-intercept
            X = 0
            c = eval(expression)
            st.write(f"Y-intercept = {c}")
            fig.add_trace(go.Scatter(
                x=[0], y=[c],
                mode="markers",
                marker=dict(color='black', size=10, symbol='star'),
                name=f"Y-intercept of func-{i+1}"
            ))

        except Exception as e:
            st.error(f"Error evaluating expression: {e}")

# Optional point marking
mark_points = st.radio(
    "Do you want to check the value of the function at specific points?",
    ["No", "Yes"],
    index=0
)

if mark_points == "Yes":
    num_points = st.number_input("Enter the number of points you want to mark:", min_value=1, step=1)
    expression_track = st.text_input("Enter the function you want to track (as a function of 'X'):")
    

    for i in range(num_points):
        
        x_pt = st.number_input(f"Enter the X-value of point {i+1}:", key=f"xpt_{i}")
        if expression_track:
            try:
                X = x_pt
                y_pt = eval(expression_track)
                st.write(f"Point-{i+1}: ({x_pt}, {y_pt:.3f})")
                color_pt = st.radio(
                    f"Choose the color of point {i+1}:",
                    ["red", "blue", "cyan", "green", "yellow", "indigo", "teal"],
                    key=f"color_{i}"
                )
                fig.add_trace(go.Scatter(
                    x=[x_pt], y=[y_pt],
                    mode='markers',
                    marker=dict(color=color_pt, size=10, symbol='star')
                ))
            except Exception as e:
                st.error(f"Invalid expression or X-value: {e}")

# Add axes lines
fig.add_hline(y=0, line_color='black')
fig.add_vline(x=0, line_color='black')

# Layout styling
fig.update_layout(
    title=dict(text=title_chart, font=dict(size=title_font_size, color='black')),
    xaxis_title=x_name,
    yaxis_title=y_name,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(x=0, y=1)
)

if st.button("Generate 2-D plot"):
    st.plotly_chart(fig, use_container_width=True)
