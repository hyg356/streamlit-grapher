import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from io import BytesIO

st.title("Scatter Plot and Fit")
st.write("Refer to the user manual in the 'home page', for any querries")

inp_type = st.radio("Choose your input type", ["Manual", "Excel", "CSV"])

#manual input

if inp_type == "Manual":
    title_inp = st.text_input("Enter title: ")
    x_mark = st.text_input("Enter label for the X-axis: ")
    y_mark = st.text_input("Enter label for the Y-axis: ")

    num_points = int(
        st.number_input("Enter the number of points: ", 
                        min_value=0, max_value=100, step=1)
    )

    X_val = []
    Y_val = []

    for i in range(num_points):
        X_val.append(
            st.number_input(
                f"Enter the X-Coordinate of point {i+1}: ",
                max_value=10000.0, min_value=-10000.0,
                step=0.01, value=0.0, key=f"manual_x_{i}"
            )
        )
        Y_val.append(
            st.number_input(
                f"Enter the Y-Coordinate of point {i+1}: ",
                max_value=10000.0, min_value=-10000.0,
                step=0.01, value=0.0, key=f"manual_y_{i}"
            )
        )

    is_best_fit = st.radio("Do you want a best fit to the points?", ["Yes", "No"])

    plt.clf()
    plt.figure(figsize=(10, 6))

    if is_best_fit == "No":
        plt.scatter(X_val, Y_val, color="red")
        plt.axvline(x=0, color='black')
        plt.axhline(y=0, color='black')
        plt.title(title_inp)
        plt.xlabel(x_mark)
        plt.ylabel(y_mark)
        plt.grid(True)
        st.pyplot(plt)

    
    if is_best_fit == "Yes":
        fit_type = st.radio(
            "Choose among the following types of fit:", 
            ["Smart fit", "Linear fit"]
        )

        # linear fit
        if fit_type == "Linear fit":
            X_data = np.array(X_val)
            Y_data = np.array(Y_val)

            m, c = np.polyfit(X_data, Y_data, 1)
            Y_pred = m * X_data + c

            plt.title(title_inp)
            plt.xlabel(x_mark)
            plt.ylabel(y_mark)
            plt.scatter(X_data, Y_data)
            plt.plot(X_data, Y_pred, label="Best Fit Line")
            plt.figtext(0.5, 0.2, f"Slope = {m:.4f}\nIntercept = {c:.4f}")
            plt.grid(True)
            plt.legend()
            st.pyplot(plt)

        # smart fit
        elif fit_type == "Smart fit":

            domain_start = st.number_input("Enter lower bound of the domain: ")
            domain_end = st.number_input("Enter upper bound of the domain: ")
            step_size = 0.1

            if st.button("Run Model"):
                X = np.array(X_val)
                y = np.array(Y_val)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                best_degree = 0
                best_error = 1000000000000000000000000.0
                best_coeff = 0

                # test polynomial degrees 1 to 5
                for d in [1, 2, 3, 4, 5]:
                    try:
                        coeff = np.polyfit(X_train, y_train, d)
                        poly = np.poly1d(coeff)
                        preds = poly(X_test)
                        error = np.mean((preds - y_test)**2)

                        if error < best_error:
                            best_error = error
                            best_degree = d
                            best_coeff = coeff

                    except:
                        pass

                poly_func = np.poly1d(best_coeff)

                arr_x = np.arange(domain_start, domain_end + step_size, step_size)
                arr_y = poly_func(arr_x)

                arr_func_eqn = []
                for i, coeff in enumerate(best_coeff):
                    if round(coeff, 2) != 0:
                        arr_func_eqn.append(f"y={round(coeff,2)}*x^{best_degree-i}")

                plt.figure(figsize=(10, 6))
                st.write(f"func: {arr_func_eqn}")
                plt.plot(arr_x, arr_y, label=f"Best Curve (degree={best_degree})")
                plt.scatter(X, y, color="red", label="Points")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt)

#excel input

elif inp_type == "Excel":
    st.write("Note: Ensure you upload a complete Excel file")
    st.title("📘 Excel Column Selector")

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        st.write("Available columns:", df.columns.tolist())
        X_header = st.text_input("Select the column for X-values:")
        Y_header = st.text_input("Select the column for Y-values:")

        if X_header in df.columns and Y_header in df.columns:
            X_val = df[X_header].values
            Y_val = df[Y_header].values

            domain_start = st.number_input("Enter lower bound of the domain: ")
            domain_end = st.number_input("Enter upper bound of the domain: ")
            step_size = 0.1

            if st.button("Run Model"):
                X = np.array(X_val)
                y = np.array(Y_val)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                best_degree = 0
                best_error = 10000000000000000000.0
                best_coeff = 0

                for d in [1, 2, 3, 4, 5]:
                    try:
                        coeff = np.polyfit(X_train, y_train, d)
                        poly = np.poly1d(coeff)
                        preds = poly(X_test)
                        error = np.mean((preds - y_test)**2)

                        if error < best_error:
                            best_error = error
                            best_degree = d
                            best_coeff = coeff

                    except:
                        pass

                poly_func = np.poly1d(best_coeff)
                arr_x = np.arange(domain_start, domain_end + step_size, step_size)
                arr_y = poly_func(arr_x)

                arr_func_eqn = []
                for i, coeff in enumerate(best_coeff):
                    if round(coeff, 2) != 0:
                        arr_func_eqn.append(f"y={round(coeff,2)}*x^{best_degree-i}")

                plt.figure(figsize=(10, 6))
                st.write(f"func: {arr_func_eqn}")
                plt.plot(arr_x, arr_y, label=f"Best Curve (degree={best_degree})")
                plt.scatter(X, y, color="red", label="Points")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt)

#CSV input

elif inp_type == "CSV":
    st.write("Note: Ensure you upload a complete CSV file")
    st.title("📘 CSV Column Selector")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("Available columns:", df.columns.tolist())
        X_header = st.text_input("Select the column for X-values:")
        Y_header = st.text_input("Select the column for Y-values:")

        if X_header in df.columns and Y_header in df.columns:
            X_val = df[X_header].values
            Y_val = df[Y_header].values

            domain_start = st.number_input("Enter lower bound of the domain: ")
            domain_end = st.number_input("Enter upper bound of the domain: ")
            step_size = 0.1

            if st.button("Run Model"):
                X = np.array(X_val)
                y = np.array(Y_val)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                best_degree = 0
                best_error = 100000000000000000000000000.0
                best_coeff = 0

                for d in [1, 2, 3, 4, 5]:
                    try:
                        coeff = np.polyfit(X_train, y_train, d)
                        poly = np.poly1d(coeff)
                        preds = poly(X_test)
                        error = np.mean((preds - y_test)**2)

                        if error < best_error:
                            best_error = error
                            best_degree = d
                            best_coeff = coeff

                    except:
                        pass

                poly_func = np.poly1d(best_coeff)
                arr_x = np.arange(domain_start, domain_end + step_size, step_size)
                arr_y = poly_func(arr_x)

                arr_func_eqn = []
                for i, coeff in enumerate(best_coeff):
                    if round(coeff, 2) != 0:
                        arr_func_eqn.append(f"y={round(coeff,2)}*x^{best_degree-i}")

                plt.figure(figsize=(10, 6))
                st.write(f"func: {arr_func_eqn}")
                plt.plot(arr_x, arr_y, label=f"Best Curve (degree={best_degree})")
                plt.scatter(X, y, color="red", label="Points")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt)
