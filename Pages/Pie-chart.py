import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("Pie-Chart")


st.write("Refer to the user manual in the 'home page', for any querries")

title_inp = st.text_input("Enter title: ")
font_size_title=st.number_input("Enter font size of the title: ", min_value=5, max_value=40, step=1)
n = st.number_input("Enter the number of items: ", min_value=1, step=1)
names = []
val = []

for i in range(int(n)):
    if i == int(n) - 1:
        add_val = sum(val)
        names.append(st.text_input(f"Enter the name of item-{i+1}: "))
        val.append(float(100 - add_val))
    else:
        names.append(st.text_input(f"Enter the name of item-{i+1}: "))
        val.append(st.number_input(f"Enter percentage of item-{i+1}: ", min_value=0.0, max_value=float(100 - sum(val))))

fig, ax = plt.subplots()
ax.pie(val, labels=names, autopct="%1.1f%%")
ax.set_title(title_inp, fontweight='bold', fontsize=font_size_title)

if st.button("Generate Pie-Chart"):
    st.pyplot(fig)


