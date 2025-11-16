import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
st.title("Bar Graph Plotter")



st.write("Refer to the user manual in the 'home page', for any querries")

upload_format = st.radio(
    "Please select the format you would like to upload data from",
    ["Excel sheet", "CSV format", "Manual entry"]
)


if upload_format == "Manual entry":
    bar_labels = []
    bar_votes = []
    
    title_inp = st.text_input("Enter title: ")
    X_label=st.text_input("Enter label for the X-axis: ")
    Y_label=st.text_input("Enter label for the Y_axis: ")

    num_points = st.number_input(
        "Enter number of labels/categories in the x-axis of your bar plot:",
        min_value=1,
        step=1
    )


    for i in range(int(num_points)):
        bar_labels.append(
            st.text_input(
                f"Enter label-{i+1}/category-{i+1} (X-axis of your plot):",
                key=f"label_{i}"
            )
        )
        bar_votes.append(
            st.number_input(
                f"Enter value (Y-axis) for label-{i+1}:",
                min_value=0.0,
                key=f"value_{i}"
            )
        )

   
    df = pd.DataFrame({
        'bar_labels': bar_labels,
        'bar_votes': bar_votes
    })

    X_val_bar = df['bar_labels']
    Y_val_bar = df['bar_votes']

    

   

   
    if st.button("Generate Bar Graph"):
        plt.clf()
        plt.title(title_inp)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.bar(X_val_bar, Y_val_bar)
        st.pyplot(plt)



elif upload_format=="Excel sheet":
    st.write("Note: Ensure you upload a complete excel file")
  

    st.title("📘 Excel Column Selector")


    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
    if uploaded_file:
        title_inp = st.text_input("Enter title: ")
        X_label=st.text_input("Enter label for the X-axis: ")
        Y_label=st.text_input("Enter label for the Y_axis: ")
        df=pd.read_excel(uploaded_file)
        st.write("Available columns:", df.columns.tolist())
        X_val=st.text_input("Select the header from the spreadsheet you would like to take as the X-axis: ")
        Y_val=st.text_input("Select the header from the spreadsheet you would like to take as the Y-axis: ")
        
        

    
        if st.button("Generate Bar plot"):
            plt.clf()
            plt.title(title_inp)
            plt.xlabel(X_label)
            plt.ylabel(Y_label)
            plt.bar(df[X_val],df[Y_val])
            st.pyplot(plt)

elif upload_format=="CSV format":
    st.write("Note: Ensure you upload a complete CSV file")
  

    st.title("📘 CSV Column Selector")


    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        title_inp = st.text_input("Enter title: ")
        X_label=st.text_input("Enter label for the X-axis: ")
        Y_label=st.text_input("Enter label for the Y_axis: ")
        df=pd.read_csv(uploaded_file)
        st.write("Available columns:", df.columns.tolist())

        X_val=st.text_input("Select the header from the spreadsheet you would like to take as the X-axis: ",[df.columns.tolist()])
        Y_val=st.text_input("Select the header from the spreadsheet you would like to take as the Y-axis: ",[df.columns.tolist()])

        
        

    
        if st.button("Generate Bar plot"):
            plt.clf()
            plt.title(title_inp)
            plt.xlabel(X_label)
            plt.ylabel(Y_label)
            plt.bar(df[X_val],df[Y_val])
            st.pyplot(plt)
    





