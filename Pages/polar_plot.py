import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.title("Interactive Polar Plotter")

st.write("Refer to the user manual in the 'Home Page' for any queries.")


# Title and labels
title = st.text_input("Enter chart title:")
title_font = st.number_input("Enter font size of the title:", min_value=5, max_value=40, step=1)
num_curves = st.number_input("Enter number of curves:", min_value=1, step=1)

fig = go.Figure()

for i in range(num_curves):
    st.write("NOTE — use prefix 'np.' for math functions (e.g., np.sin, np.cos, np.pi)")
    expression = st.text_input(f"Enter polar function r = f(theta) for curve {i+1}:")
    domain_lower = st.number_input(f"Enter lower bound of theta for curve {i+1}:", -10*np.pi, 10*np.pi, 0.0)
    domain_upper = st.number_input(f"Enter upper bound of theta for curve {i+1}:", -10*np.pi, 10*np.pi, 2*np.pi)

    if expression:
        try:
            theta = np.linspace(domain_lower, domain_upper, 1000)
            r = eval(expression)
            r = np.where(r < 0, np.nan, r)
            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=np.degrees(theta),  # convert radians to degrees for Plotly
                mode='lines',
                name=f"Curve {i+1}: r={expression}"
            ))
        except Exception as e:
            st.error(f"Error evaluating expression: {e}")

fig.update_layout(
    title=dict(text=title, font=dict(size=title_font)),
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=True
)

if st.button("Generate Polar Plot"):
    st.plotly_chart(fig)



