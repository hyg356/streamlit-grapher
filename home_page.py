import streamlit as st


st.image("plot_twist.png", use_container_width=True)

st.title("Welcome to Plot Twist!")
st.write("Choose a plot from the pages on the left.")


















st.page_link("pages/Scatter_and_line.py", label="Scatter and Line plot")
st.page_link("pages/bar_plot.py", label="Bar")
st.page_link("pages/histogram_plot.py", label="Histogram")
st.page_link("pages/polar_plot.py", label="Polar plot")
st.page_link("pages/_3D-plot.py", label=" 3D function plot")
st.page_link("pages/_2D-plot.py", label="2D function plot")
st.page_link("pages/Pie-chart.py", label="Pie-Chart")

with st.expander("▼ Click to view user manual"):
    st.title("1) Input rules")
    st.write("Ensure that you stick to the exact input type as specified above, as well as upper/lower case")
    st.write("use the prefix 'np.' while plugging in mathematical constants. In case you experience a glitch, plug in the approximate value of the constant")
    st.write("Example: ")
    st.write("e= np.e")
    st.write("pi= np.pi")

    st.title("2) Mathematics operation rules: ")

    st.write("The following are the syntaxes for some common mathematical operations: ")
    st.write("Multiplication: *")
    st.write("Addition: +")
    st.write("Subtraction: -")
    st.write("Division: /")
    st.write("2X= 2*X; ENSURE THAT YOU DO NOT FORGET THE MULTIPLICATION SIGN!!")
    st.write("Exponentiation(a^b, or 'a' power 'b'): a ** b")

    st.write("The following are the syntaxes for some mathematical functions: ")
    st.write("NOTE: Ensure you use the prefix 'np.' even for the mathematical functions not listed below")
    st.write("The following are a few examples: ")
    st.write("a^(1/2) or sqrt(a)= np.sqrt(a)")
    st.write("sin(a)= np.sin(a)")

