import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================
# APP HEADER
# =========================
st.title("Histogram Graph Plotter")
st.write("Refer to the user manual in the 'home page', for any queries")

chose_upload = st.radio(
    "Choose your upload format:",
    ["Manual Input", "CSV file"]
)

# =========================
# MANUAL INPUT MODE
# =========================
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

# =========================
# CSV MODE
# =========================
elif chose_upload == "CSV file":

    st.write("Note: Ensure you upload a complete CSV file")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:

        model_choice = st.radio("Enter model:", ["Curve based", "AI driven"])

        title_inp = st.text_input("Enter title:")
        X_label = st.text_input("Enter label for X-axis:")
        Y_label = st.text_input("Enter label for Y-axis:")

        df = pd.read_csv(uploaded_file)
        st.write("Available columns:", df.columns.tolist())
        marks = st.text_input("Select the header to use:")

        if marks not in df.columns:
            st.warning("Please enter a valid column name.")
            st.stop()

        max_val = int(st.number_input("Enter max marks:", min_value=1, step=1))

        # =========================
        # CLEAN MARKS
        # =========================
        marks_array = pd.to_numeric(df[marks], errors="coerce").dropna().to_numpy()
        sorted_marks = np.sort(marks_array)

        median_marks = sorted_marks[len(sorted_marks) // 2]
        NC_cut = 0
        E_cut = round(0.30 * median_marks, 2)

        # =========================
        # CURVE BASED MODEL
        # =========================
        if model_choice == "Curve based":

             X_val = [-1]
             Y_val = [0]

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

        # ---- END CURVE: ONE MORE THAN PREVIOUS ELEMENT ----
             X_val.append(X_val[-1] + 1)
             Y_val.append(0)

        # -------- AREA COMPUTATION --------
             area_arr = []

             area_1 = 0.5 * (X_val[1] - X_val[0]) * Y_val[1]

             for j in range(1, len(X_val) - 1):
                 x1, x2 = X_val[j], X_val[j + 1]
                 y1, y2 = Y_val[j], Y_val[j + 1]
                 base = x2 - x1

                 if y1 == y2:
                     area_arr.append(base * y1)
                 else:
                     rect = min(y1, y2) * base
                     tri = 0.5 * base * abs(y2 - y1)
                     area_arr.append(rect + tri)

             area_2 = 0.5 * (X_val[-1] - X_val[-2]) * Y_val[-2]
             total_area = area_1 + sum(area_arr) + area_2

        # -------- AREA TARGETS --------
             A_cut_area = 0.85 * total_area
             A_minus_cut_area = 0.70 * total_area
             B_cut_area = 0.55 * total_area
             B_minus_cut_area = 0.40 * total_area
             C_cut_area = 0.25 * total_area
             C_minus_cut_area = 0.15 * total_area
             D_cut_area = 0.08 * total_area
             # -------- FIND CUTS FROM AREA --------
             sum_area = area_1

             D_cut = C_minus_cut = C_cut = B_minus_cut = B_cut = A_minus_cut = A_cut = None

             for q in range(len(area_arr)):
                 sum_area += area_arr[q]
                 x_mid = (X_val[q + 1] + X_val[q + 2]) / 2

                 if D_cut is None and sum_area >= D_cut_area:
                     D_cut = x_mid
                 if C_minus_cut is None and sum_area >= C_minus_cut_area:
                     C_minus_cut = x_mid
                 if C_cut is None and sum_area >= C_cut_area:
                     C_cut = x_mid
                 if B_minus_cut is None and sum_area >= B_minus_cut_area:
                     B_minus_cut = x_mid
                 if B_cut is None and sum_area >= B_cut_area:
                      B_cut = x_mid
                 if A_minus_cut is None and sum_area >= A_minus_cut_area:
                     A_minus_cut = x_mid
                 if A_cut is None and sum_area >= A_cut_area:
                   A_cut = x_mid


        # =========================
        # AI DRIVEN MODEL
        # =========================
        else:

            passing = sorted_marks[sorted_marks >= E_cut].reshape(-1, 1)

            N_GRADES = 8
            if len(passing) < N_GRADES:
                st.error("Not enough students for AI-driven grading.")
                st.stop()

            kmeans = KMeans(
                n_clusters=N_GRADES,
                random_state=42,
                n_init=20
            )
            kmeans.fit(passing)

            centers = np.sort(kmeans.cluster_centers_.flatten())
            cuts = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

            D_cut, C_minus_cut, C_cut, B_minus_cut, B_cut, A_minus_cut, A_cut = cuts

            def fix(x, lo):
                return lo if x <= lo else x

            D_cut = fix(D_cut, E_cut + 1)
            C_minus_cut = fix(C_minus_cut, D_cut + 1)
            C_cut = fix(C_cut, C_minus_cut + 1)
            B_minus_cut = fix(B_minus_cut, C_cut + 1)
            B_cut = fix(B_cut, B_minus_cut + 1)
            A_minus_cut = fix(A_minus_cut, B_cut + 1)
            A_cut = fix(A_cut, A_minus_cut + 1)

        # =========================
        # SLIDERS (NO NC SLIDER)
        # =========================
        E_cut = st.slider("E", 0.0, float(max_val), float(E_cut), 1.0)
        D_cut = st.slider("D", E_cut + 1, float(max_val), float(D_cut), 1.0)
        C_minus_cut = st.slider("C-", D_cut + 1, float(max_val), float(C_minus_cut), 1.0)
        C_cut = st.slider("C", C_minus_cut + 1, float(max_val), float(C_cut), 1.0)
        B_minus_cut = st.slider("B-", C_cut + 1, float(max_val), float(B_minus_cut), 1.0)
        B_cut = st.slider("B", B_minus_cut + 1, float(max_val), float(B_cut), 1.0)
        A_minus_cut = st.slider("A-", B_cut + 1, float(max_val), float(A_minus_cut), 1.0)
        A_cut = st.slider("A", A_minus_cut + 1, float(max_val), float(A_cut), 1.0)

        # =========================
        # ASSIGN GRADES
        # =========================
        df.drop(columns=["Grade"], errors="ignore", inplace=True)

        df["Grade"] = df[marks].apply(
            lambda x:
                "A" if x > A_cut else
                "A-" if x > A_minus_cut else
                "B" if x > B_cut else
                "B-" if x > B_minus_cut else
                "C" if x > C_cut else
                "C-" if x > C_minus_cut else
                "D" if x > D_cut else
                "E" if x > E_cut else
                "NC"
        )

        # =========================
        # PLOT + DOWNLOAD
        # =========================
        if st.button("Generate Histogram plot"):
            plt.figure(figsize=(16, 8))
            plt.title(title_inp)
            plt.xlabel(X_label)
            plt.ylabel(Y_label)

            for cut in [A_cut, A_minus_cut, B_cut, B_minus_cut,
                        C_cut, C_minus_cut, D_cut, E_cut]:
                plt.axvline(cut)

            bins = np.arange(0, max_val + 2, 1)
            plt.hist(df[marks], bins=bins, edgecolor="black")
            plt.xlim(0, max_val + 1)

            st.pyplot(plt)
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Updated CSV",
                csv,
                "updated_file.csv",
                "text/csv"
            )
