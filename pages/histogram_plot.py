import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("Histogram Graph Plotter")

st.write("Refer to the user manual in the 'home page', for any querries")

chose_upload = st.radio(
    "Choose your upload format:",
    ["Manual Input", "CSV file"]
)



if chose_upload == "Manual Input":
    title_inp = st.text_input("Enter title:")
    X_label = st.text_input("Enter X-axis label:")
    Y_label = st.text_input("Enter Y-axis label:")

    num_input = st.number_input("Enter the number of points:", min_value=1, step=1)
    num_bins = st.number_input("Enter number of bins:", min_value=1, step=1)

    val = []
    for i in range(int(num_input)):
        val.append(
            st.number_input(f"Enter value-{i+1}:", min_value=0.0, key=f"val_{i}")
        )

    is_cumm = st.radio("Do you want a cumulative plot?", ["Yes", "No"])

    if st.button("Generate Histogram plot"):
        plt.clf()
        plt.title(title_inp)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.hist(val, bins=int(num_bins), cumulative=(is_cumm == "Yes"))
        st.pyplot(plt)



elif chose_upload == "CSV file":
    st.write("Note: Ensure you upload a complete CSV file")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        title_inp = st.text_input("Enter title: ")
        X_label = st.text_input("Enter label for X-axis: ")
        Y_label = st.text_input("Enter label for Y-axis: ")
        num_bins = st.number_input("Enter number of bins: ", min_value=1, step=1)

        df = pd.read_csv(uploaded_file)
        st.write("Available columns:", df.columns.tolist())
        marks = st.text_input("Select the header from the spreadsheet you would like to use: ")

        max_val = st.number_input("Enter max marks:", min_value=1, step=1)
        max_val = int(max_val)

        min_val = 0

        Course_code = st.text_input("Enter course code: ")
        Course_name = st.text_input("Enter course name: ")

        marks_array = pd.to_numeric(df[marks], errors='coerce').dropna().to_numpy()

        sorted_marks = np.sort(marks_array)

        X_val = []
        Y_val = []

        X_val.append(-1)
        Y_val.append(0)

        n = len(sorted_marks)
        i = 0

        while i < n:
            temp = sorted_marks[i]
            count = 0

            while i < n and sorted_marks[i] == temp:
                count += 1
                i += 1

            X_val.append(temp)
            Y_val.append(count)

            if i < n:
                temp_2 = sorted_marks[i]
                if temp_2 - temp > 1 and len(Y_val) >= 2:
                    X_val.append((temp + temp_2) / 2)
                    Y_val.append(0.5 * ((Y_val[-1] + Y_val[-2]) / 2))

        X_val.append(max_val + 1)
        Y_val.append(0)

        area_arr = []

        base = X_val[1] - X_val[0]
        h1 = Y_val[1]
        area_arr.append(0.5 * base * h1)

        for j in range(1, len(X_val) - 1):
            x1 = X_val[j]
            x2 = X_val[j + 1]
            y1 = Y_val[j]
            y2 = Y_val[j + 1]
            base = x2 - x1

            if y1 == y2:
                area_arr.append(base * y1)
            else:
                rect = min(y1, y2) * base
                tri = 0.5 * base * abs(y2 - y1)
                area_arr.append(rect + tri)

        base = X_val[-1] - X_val[-2]
        height = Y_val[-2]
        area_arr.append(0.5 * base * height)

        total_area = sum(area_arr)

        A_cut_area = 5 * (total_area / 6)
        A_minus_cut_area = 4 * (total_area / 6)
        B_cut_area = 3 * (total_area / 6)
        B_minus_cut_area = 2 * (total_area / 6)
        C_cut_area = total_area / 6

        median_student_index = len(sorted_marks) // 2
        median_marks = sorted_marks[median_student_index]

        sum_area = 0
        C_cut = 0
        B_minus_cut = 0
        B_cut = 0
        A_minus_cut = 0
        A_cut = 0

        C_minus_cut = 0.3 * median_marks
        NC_cut = 0

        for q in range(len(area_arr)):
            sum_area += area_arr[q]

            if C_cut == 0 and sum_area >= C_cut_area:
                C_cut = X_val[q]
            if B_minus_cut == 0 and sum_area >= B_minus_cut_area:
                B_minus_cut = X_val[q]
            if B_cut == 0 and sum_area >= B_cut_area:
                B_cut = X_val[q]
            if A_minus_cut == 0 and sum_area >= A_minus_cut_area:
                A_minus_cut = X_val[q]
            if A_cut == 0 and sum_area >= A_cut_area:
                A_cut = X_val[q]

        NC_cut = st.slider("NC", min_value=0.0, max_value=float(max_val), value=0.0, step=1.0)
        C_minus_cut = st.slider("C-", min_value=float(NC_cut + 1), max_value=float(max_val),
                                value=float(C_minus_cut), step=1.0)
        C_cut = st.slider("C", min_value=float(C_minus_cut + 1), max_value=float(max_val),
                          value=float(C_cut), step=1.0)
        B_minus_cut = st.slider("B-", min_value=float(C_cut + 1), max_value=float(max_val),
                                value=float(B_minus_cut), step=1.0)
        B_cut = st.slider("B", min_value=float(B_minus_cut + 1), max_value=float(max_val),
                          value=float(B_cut), step=1.0)
        A_minus_cut = st.slider("A-", min_value=float(B_cut + 1), max_value=float(max_val),
                                value=float(A_minus_cut), step=1.0)
        A_cut = st.slider("A", min_value=float(A_minus_cut + 1), max_value=float(max_val),
                          value=float(A_cut), step=1.0)

        A_count = A_minus_count = B_count = B_minus_count = C_count = C_minus_count = NC_count = 0

        for num in df[marks]:
            if num > A_cut:
                A_count += 1
            elif num > A_minus_cut:
                A_minus_count += 1
            elif num > B_cut:
                B_count += 1
            elif num > B_minus_cut:
                B_minus_count += 1
            elif num > C_cut:
                C_count += 1
            elif num > C_minus_cut:
                C_minus_count += 1
            else:
                NC_count += 1

        
        if st.button("Generate Histogram plot"):

            Average = round(df[marks].mean(), 2)

            sum_grade = 0
            for ch in df["Grade"] if "Grade" in df else []:
                pass  # placeholder

           
            st.write(f"Average = {Average}")
            st.write(f"A cutoff={A_cut};  A : {A_count}")
            st.write(f"A- cutoff={A_minus_cut};  A- : {A_minus_count}")
            st.write(f"B cutoff={B_cut};  B : {B_count}")
            st.write(f"B- cutoff={B_minus_cut};  B- : {B_minus_count}")
            st.write(f"C cutoff={C_cut};  C : {C_count}")
            st.write(f"C- cutoff={C_minus_cut};  C- : {C_minus_count}")
            st.write(f"NC cutoff={NC_cut}; NC : {NC_count}")

            plt.clf()
            plt.figure(figsize=(16, 8))
            plt.title(title_inp)
            plt.xlabel(X_label)
            plt.ylabel(Y_label)

            plt.axvline(A_cut, color='red')
            plt.axvline(A_minus_cut, color='red')
            plt.axvline(B_cut, color='red')
            plt.axvline(Average, color='blue')
            plt.axvline(B_minus_cut, color='red')
            plt.axvline(C_cut, color='red')
            plt.axvline(C_minus_cut, color='red')
            plt.axvline(NC_cut, color="black")

            bins = np.linspace(0, max_val + 1.0, int(num_bins))   # note +1.0 (not 1e-6)
            plt.hist(df[marks], bins=bins, edgecolor='black')
            plt.xlim(0, max_val + 1.0)
            plt.xticks(range(0, int(max_val)+1, 10), fontsize=12)


    

            st.write("Note- ensure that a 'Grade' column doesnt exist in your spreadsheet")

            df["Grade"] = df[marks].apply(
                lambda x: (
                    'A' if x > A_cut else
                    'A-' if x > A_minus_cut else
                    'B' if x > B_cut else
                    'B-' if x > B_minus_cut else
                    'C' if x > C_cut else
                    'C-' if x > C_minus_cut else
                    'NC'
                )
            )

            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Updated CSV", csv, "updated_file.csv", "text/csv")

            sum_grade = 0
            for ch in df["Grade"]:
                if ch == "A":
                    sum_grade += 10
                elif ch == "A-":
                    sum_grade += 9
                elif ch == "B":
                    sum_grade += 8
                elif ch == "B-":
                    sum_grade += 7
                elif ch == "C":
                    sum_grade += 6
                elif ch == "C-":
                    sum_grade += 5
                elif ch == "NC":
                    sum_grade += 0

            len_grade = sum(1 for g in df["Grade"] if g != "NC")

            MGPV = round(sum_grade / len_grade, 2)
            Average = round(df[marks].mean(), 2)

            st.text(f"Average= {Average:.2f} | MGPV= {MGPV:.2f} | Course Code= {Course_code} | Course name: {Course_name}")


            st.pyplot(plt)

            
