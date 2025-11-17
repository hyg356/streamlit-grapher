import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
st.title("Histogram Graph Plotter")


st.write("Refer to the user manual in the 'home page', for any querries")

chose_upload = st.radio(
    "Choose your upload format:", 
    ["Manual Input", "CSV file", "Excel Spreadsheet"]
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


elif chose_upload=="Excel Spreadsheet":
    st.write("Note: Ensure you upload a complete Excel file")
    st.title("📘 Excel Column Selector")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

    if uploaded_file:
        title_inp=st.text_input("Enter title: ")
        X_label=st.text_input("Enter label for X_axis: ")
        Y_label=st.text_input("Enter label for Y-axis: ")
        num_bins=st.number_input("Enter number of bins: ", min_value=1, step=1)
        df=pd.read_excel(uploaded_file)
        st.write("Available columns:", df.columns.tolist())
        X_name=st.text_input("Select the header from the spreadsheet you would like to use: ")
        

        if X_name in df.columns:
           min_val = df[X_name].min()
           max_val = df[X_name].max()
        else:
           min_val = 0
           max_val = 100

        
        is_bound=st.radio("Do you want the grade boundaries to be displayed(interactive adjustable boundaries)?", ["Yes","No"])
        

        # non sd-method for small data sets
        if is_bound=="Yes" and df[X_name].count()<20:

            avg = df[X_name].mean()
            len_score = df[X_name].count()

            arr_score_up = []
            arr_score_down = []

            # SPLIT ABOVE / BELOW AVERAGE
            for i in range(len_score):
                if df[X_name].iloc[i] >= avg:
                    arr_score_up.append(df[X_name].iloc[i])
                else:
                    arr_score_down.append(df[X_name].iloc[i])

            arr_score_up.sort()
            arr_score_down.sort()

            # SCORE DIFFS
            arr_dif_up = []
            arr_dif_down = []

            for k in range(len(arr_score_up) - 1):
                arr_dif_up.append(arr_score_up[k+1] - arr_score_up[k])

            for k in range(len(arr_score_down) - 1):
                arr_dif_down.append(arr_score_down[k+1] - arr_score_down[k])

            if len(arr_dif_up)==0: arr_dif_up=[0]
            if len(arr_dif_down)==0: arr_dif_down=[0]

            # initialize max values
            dif_up_1 = arr_dif_up[0]
            dif_up_2 = arr_dif_up[0]
            dif_up_3 = arr_dif_up[0]

            dif_down_1 = arr_dif_down[0]
            dif_down_2 = arr_dif_down[0]

            count_in_up_1 = 0
            count_in_up_2 = 0
            count_in_up_3 = 0

            count_in_down_1 = 0
            count_in_down_2 = 0

            # UP SIDE largest gap
            for i in range(len(arr_dif_up)):
                if arr_dif_up[i] > dif_up_1:
                    dif_up_1 = arr_dif_up[i]
                    count_in_up_1 = i

            # second largest
            for i in range(len(arr_dif_up)):
                if arr_dif_up[i] > dif_up_2 and arr_dif_up[i] < dif_up_1:
                    dif_up_2 = arr_dif_up[i]
                    count_in_up_2 = i

            # third largest
            for i in range(len(arr_dif_up)):
                if arr_dif_up[i] > dif_up_3 and arr_dif_up[i] < dif_up_2:
                    dif_up_3 = arr_dif_up[i]
                    count_in_up_3 = i

            # DOWN SIDE
            for i in range(len(arr_dif_down)):
                if arr_dif_down[i] > dif_down_1:
                    dif_down_1 = arr_dif_down[i]
                    count_in_down_1 = i

            for i in range(len(arr_dif_down)):
                if arr_dif_down[i] > dif_down_2 and arr_dif_down[i] < dif_down_1:
                    dif_down_2 = arr_dif_down[i]
                    count_in_down_2 = i

            # BACKTRACK CUTS
            dif_1_up_val_cut = 0
            dif_2_up_val_cut = 0
            dif_3_up_val_cut = 0

            dif_1_down_val_cut = 0
            dif_2_down_val_cut = 0

            # ABOVE AVERAGE CUTS
            for i in range(1, len(arr_score_up)):
                gap = arr_score_up[i] - arr_score_up[i-1]
                if gap == dif_up_1 and i-1 == count_in_up_1:
                    dif_1_up_val_cut = arr_score_up[i]
                elif gap == dif_up_2 and i-1 == count_in_up_2:
                    dif_2_up_val_cut = arr_score_up[i]
                elif gap == dif_up_3 and i-1 == count_in_up_3:
                    dif_3_up_val_cut = arr_score_up[i]

            cuts_up = sorted([dif_1_up_val_cut, dif_2_up_val_cut, dif_3_up_val_cut], reverse=True)
            A_cut = cuts_up[0]
            A_minus_cut = cuts_up[1]
            B_cut = cuts_up[2]

            # BELOW AVERAGE CUTS
            for i in range(1, len(arr_score_down)):
                gap = arr_score_down[i] - arr_score_down[i-1]
                if gap == dif_down_1 and i-1 == count_in_down_1:
                    dif_1_down_val_cut = arr_score_down[i]
                elif gap == dif_down_2 and i-1 == count_in_down_2:
                    dif_2_down_val_cut = arr_score_down[i]

            if dif_1_down_val_cut > dif_2_down_val_cut:
                C_cut = dif_1_down_val_cut
                C_minus_cut = dif_2_down_val_cut
            else:
                C_cut = dif_2_down_val_cut
                C_minus_cut = dif_1_down_val_cut
            



        elif is_bound=="Yes" and df[X_name].count()>=20:

            score_up=[]
            score_down=[]
            avg=df[X_name].mean()

            for num in df[X_name]:
                if num>avg:
                    score_up.append(num)
                if num<avg:
                    score_down.append(num)

            md_up=[]
            md_down=[]

            for num in score_up:
                md_up.append(num-avg)
            for num in score_down:
                md_down.append(avg-num)

            md_up = sorted(md_up)
            md_down = sorted(md_down)

            md_dif_up=[]
            md_dif_down=[]

            for i in range(1,len(md_up)):
                md_dif_up.append(md_up[i]-md_up[i-1])
            for j in range(1,len(md_down)):
                md_dif_down.append(md_down[j]-md_down[j-1])

            md_dif_up = sorted(md_dif_up, reverse=True)
            md_dif_down = sorted(md_dif_down, reverse=True)

            md_dif_up_1=md_dif_up[0]
            md_dif_up_2=md_dif_up[1]
            md_dif_up_3=md_dif_up[2]

            md_dif_down_1=md_dif_down[0]
            md_dif_down_2=md_dif_down[1]

            bound_up_1=0
            bound_up_1_is_filled=False
            bound_up_2=0
            bound_up_2_is_filled=False
            bound_up_3=0
            bound_up_3_is_filled=False

            bound_down_1=0
            bound_down_1_is_filled=False
            bound_down_2=0
            bound_down_2_is_filled=False

            for i in range(1,len(md_up)):
                if md_up[i]-md_up[i-1]==md_dif_up_1 and not bound_up_1_is_filled:
                    bound_up_1=md_up[i-1]
                    bound_up_1_is_filled=True
                elif md_up[i]-md_up[i-1]==md_dif_up_2 and not bound_up_2_is_filled:
                    bound_up_2=md_up[i-1]
                    bound_up_2_is_filled=True
                elif md_up[i]-md_up[i-1]==md_dif_up_3 and not bound_up_3_is_filled:
                    bound_up_3=md_up[i-1]
                    bound_up_3_is_filled=True

            arr_up=[bound_up_1,bound_up_2,bound_up_3]
            arr_up=sorted(arr_up)

            min_val=df[X_name].min()
            max_val=df[X_name].max()


          
            

            for j in range(1,len(md_down)):
                if md_down[j]-md_down[j-1]==md_dif_down_1 and not bound_down_1_is_filled:
                    bound_down_1=md_down[j-1]
                    bound_down_1_is_filled=True
                elif md_down[j]-md_down[j-1]==md_dif_down_2 and not bound_down_2_is_filled:
                    bound_down_2=md_down[j-1]
                    bound_down_2_is_filled=True

            arr_down=[bound_down_1,bound_down_2]
            arr_down=sorted(arr_down)

            NC_cut=st.slider(
            "NC",
            min_value=0.0,
            max_value=float(max_val),
            value=0.0,
            step=1.0
            )

           
            C_minus_cut = st.slider(
            "C-",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(avg - arr_down[1]-1),   
            step=1.0
            )
            C_cut = st.slider(
            "C",
            min_value=float(C_minus_cut+1),
            max_value=float(max_val),
            value=float(avg - arr_down[0]-1),   
            step=1.0
            )
            B_minus_cut = st.slider(
            "B-",
            min_value=float(C_cut+1),
            max_value=float(max_val),
            value=avg-1,   
            step=1.0
            )
            B_cut = st.slider(
            "B",
            min_value=float(B_minus_cut+1),
            max_value=float(max_val),
            value=float(avg + arr_up[0]-1),   
            step=1.0
            )
            A_minus_cut = st.slider(
            "A-",
            min_value=float(B_cut+1),
            max_value=float(max_val),
            value=float(avg + arr_up[1]-1),   
            step=1.0
            )
            A_cut = st.slider(
            "A",
            min_value=float(A_minus_cut+1),
            max_value=float(max_val),
            value=float(avg + arr_up[2]-1),   
            step=1.0
            )
            A_count=0
            A_minus_count=0
            B_count=0
            B_minus_count=0
            C_count=0
            C_minus_count=0
            NC_count=0

            for num in df[X_name]:
                if num>A_cut:
                    A_count+=1
                elif num>A_minus_cut:
                    A_minus_count+=1
                elif num>B_cut:
                    B_count+=1
                elif num>B_minus_cut:
                    B_minus_count+=1
                elif num>C_cut:
                    C_count+=1
                elif num>C_minus_cut:
                    C_minus_count+=1
                else:
                    NC_count+=1


        if st.button("Generate Histogram plot"):
            
            plt.clf()
            plt.title(title_inp)
            avg_val=round(df[X_name].mean(),2)
            st.text(f"Average={avg_val}")
            plt.xlabel(X_label)
            plt.ylabel(Y_label)

            if is_bound=="Yes":
                plt.figure(figsize=(16,8))
                plt.axvline(A_cut,color='red')
                plt.axvline(A_minus_cut, color='red')
                plt.axvline(B_cut, color='red')
                plt.axvline(avg, color='blue')
                plt.axvline(C_cut, color='red')
                plt.axvline(C_minus_cut, color='red')
                plt.axvline(NC_cut, color="black")

                st.write(f"A cutoff={A_cut};  A : {A_count}")
                st.write(f"A minus cutoff={A_minus_cut};  A- : {A_minus_count}")
                st.write(f"B cutoff={B_cut};  B : {B_count}")
                st.write(f"B- cutoff={round(avg, 2)};  B- : {B_minus_count}")
                st.write(f"C cutoff={C_cut};  C : {C_count}")
                st.write(f"C- cutoff={C_minus_count};  C- : {C_minus_count}")
                st.write(f"NC cutoff={NC_cut}; NC : {NC_count}")


                
            
            
            
            





            # Draw histogram AND get counts + bins
            
            plt.xticks(range(int(min_val), int(max_val) + 1, 3), fontsize=10)
            plt.hist(df[X_name],bins=num_bins,edgecolor='black')
            st.write("Note- ensure that a 'Grade' column doesnt exist in your spreadsheet")
            
            df["Grade"] = df[X_name].apply(
               lambda x: (
               'A'  if x > A_cut else
               'A-' if x > A_minus_cut else
               'B'  if x > B_cut else
               'B-' if x > B_minus_cut else
               'C'  if x > C_cut else
               'C-' if x > C_minus_cut else
               'NC'
               )
            )

    # SHOW UPDATED DF
            st.dataframe(df)

    # DOWNLOAD UPDATED Excel
              

            output = BytesIO()
            df.to_excel(output, index=False)
            excel_data = output.getvalue()

            st.download_button(
            label="Download Updated Excel",
            data=excel_data,
            file_name="updated_file.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
             )
            st.pyplot(plt)
        
        elif is_bound=="No" and st.button("Generate Histogram plot"):
             plt.clf()
             plt.figure(figsize=(16,8))
             plt.title(title_inp)
             plt.xlabel(X_label)
             plt.ylabel(Y_label)
             plt.hist(df[X_name], bins=num_bins, edgecolor='black')
             st.pyplot(plt)
             


        





elif chose_upload=="CSV file":
    st.write("Note: Ensure you upload a complete CSV file")
    st.title("📘 CSV Column Selector")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        title_inp=st.text_input("Enter title: ")
        X_label=st.text_input("Enter label for X_axis: ")
        Y_label=st.text_input("Enter label for Y-axis: ")
        num_bins=st.number_input("Enter number of bins: ", min_value=1, step=1)
        df=pd.read_csv(uploaded_file)
        st.write("Available columns:", df.columns.tolist())
        X_name=st.text_input("Select the header from the spreadsheet you would like to use: ")
        

       
        if X_name in df.columns:
           min_val = df[X_name].min()
           max_val = df[X_name].max()
        else:
           min_val = 0
           max_val = 100
        
        Course_code=st.text_input("Enter course code: ")
        Course_name=st.text_input("Enter course name: ")

        
        is_bound=st.radio("Do you want the grade boundaries to be displayed(interactive adjustable boundaries)?", ["Yes","No"])
        

        # non sd-method for small data sets
        if is_bound=="Yes" and df[X_name].count()<20:
            Course_code=st.text_input("Enter course code: ")

            avg = df[X_name].mean()
            len_score = df[X_name].count()

            arr_score_up = []
            arr_score_down = []

            # SPLIT ABOVE / BELOW AVERAGE
            for i in range(len_score):
                if df[X_name].iloc[i] >= avg:
                    arr_score_up.append(df[X_name].iloc[i])
                else:
                    arr_score_down.append(df[X_name].iloc[i])

            arr_score_up.sort()
            arr_score_down.sort()

            # SCORE DIFFS
            arr_dif_up = []
            arr_dif_down = []

            for k in range(len(arr_score_up) - 1):
                arr_dif_up.append(arr_score_up[k+1] - arr_score_up[k])

            for k in range(len(arr_score_down) - 1):
                arr_dif_down.append(arr_score_down[k+1] - arr_score_down[k])

            if len(arr_dif_up)==0: arr_dif_up=[0]
            if len(arr_dif_down)==0: arr_dif_down=[0]

            # initialize max values
            dif_up_1 = arr_dif_up[0]
            dif_up_2 = arr_dif_up[0]
            dif_up_3 = arr_dif_up[0]

            dif_down_1 = arr_dif_down[0]
            dif_down_2 = arr_dif_down[0]

            count_in_up_1 = 0
            count_in_up_2 = 0
            count_in_up_3 = 0

            count_in_down_1 = 0
            count_in_down_2 = 0

            # UP SIDE largest gap
            for i in range(len(arr_dif_up)):
                if arr_dif_up[i] > dif_up_1:
                    dif_up_1 = arr_dif_up[i]
                    count_in_up_1 = i

            # second largest
            for i in range(len(arr_dif_up)):
                if arr_dif_up[i] > dif_up_2 and arr_dif_up[i] < dif_up_1:
                    dif_up_2 = arr_dif_up[i]
                    count_in_up_2 = i

            # third largest
            for i in range(len(arr_dif_up)):
                if arr_dif_up[i] > dif_up_3 and arr_dif_up[i] < dif_up_2:
                    dif_up_3 = arr_dif_up[i]
                    count_in_up_3 = i

            # DOWN SIDE
            for i in range(len(arr_dif_down)):
                if arr_dif_down[i] > dif_down_1:
                    dif_down_1 = arr_dif_down[i]
                    count_in_down_1 = i

            for i in range(len(arr_dif_down)):
                if arr_dif_down[i] > dif_down_2 and arr_dif_down[i] < dif_down_1:
                    dif_down_2 = arr_dif_down[i]
                    count_in_down_2 = i

            # BACKTRACK CUTS
            dif_1_up_val_cut = 0
            dif_2_up_val_cut = 0
            dif_3_up_val_cut = 0

            dif_1_down_val_cut = 0
            dif_2_down_val_cut = 0

            # ABOVE AVERAGE CUTS
            for i in range(1, len(arr_score_up)):
                gap = arr_score_up[i] - arr_score_up[i-1]
                if gap == dif_up_1 and i-1 == count_in_up_1:
                    dif_1_up_val_cut = arr_score_up[i]
                elif gap == dif_up_2 and i-1 == count_in_up_2:
                    dif_2_up_val_cut = arr_score_up[i]
                elif gap == dif_up_3 and i-1 == count_in_up_3:
                    dif_3_up_val_cut = arr_score_up[i]

            cuts_up = sorted([dif_1_up_val_cut, dif_2_up_val_cut, dif_3_up_val_cut], reverse=True)
            A_cut = cuts_up[0]
            A_minus_cut = cuts_up[1]
            B_cut = cuts_up[2]

            # BELOW AVERAGE CUTS
            for i in range(1, len(arr_score_down)):
                gap = arr_score_down[i] - arr_score_down[i-1]
                if gap == dif_down_1 and i-1 == count_in_down_1:
                    dif_1_down_val_cut = arr_score_down[i]
                elif gap == dif_down_2 and i-1 == count_in_down_2:
                    dif_2_down_val_cut = arr_score_down[i]

            if dif_1_down_val_cut > dif_2_down_val_cut:
                C_cut = dif_1_down_val_cut
                C_minus_cut = dif_2_down_val_cut
            else:
                C_cut = dif_2_down_val_cut
                C_minus_cut = dif_1_down_val_cut


        elif is_bound=="Yes" and df[X_name].count()>=20:
            

            score_up=[]
            score_down=[]
            avg=df[X_name].mean()

            for num in df[X_name]:
                if num>avg:
                    score_up.append(num)
                if num<avg:
                    score_down.append(num)

            md_up=[]
            md_down=[]

            for num in score_up:
                md_up.append(num-avg)
            for num in score_down:
                md_down.append(avg-num)

            md_up = sorted(md_up)
            md_down = sorted(md_down)

            md_dif_up=[]
            md_dif_down=[]

            for i in range(1,len(md_up)):
                md_dif_up.append(md_up[i]-md_up[i-1])
            for j in range(1,len(md_down)):
                md_dif_down.append(md_down[j]-md_down[j-1])

            md_dif_up = sorted(md_dif_up, reverse=True)
            md_dif_down = sorted(md_dif_down, reverse=True)

            md_dif_up_1=md_dif_up[0]
            md_dif_up_2=md_dif_up[1]
            md_dif_up_3=md_dif_up[2]

            md_dif_down_1=md_dif_down[0]
            md_dif_down_2=md_dif_down[1]

            bound_up_1=0
            bound_up_1_is_filled=False
            bound_up_2=0
            bound_up_2_is_filled=False
            bound_up_3=0
            bound_up_3_is_filled=False

            bound_down_1=0
            bound_down_1_is_filled=False
            bound_down_2=0
            bound_down_2_is_filled=False

            for i in range(1,len(md_up)):
                if md_up[i]-md_up[i-1]==md_dif_up_1 and not bound_up_1_is_filled:
                    bound_up_1=md_up[i-1]
                    bound_up_1_is_filled=True
                elif md_up[i]-md_up[i-1]==md_dif_up_2 and not bound_up_2_is_filled:
                    bound_up_2=md_up[i-1]
                    bound_up_2_is_filled=True
                elif md_up[i]-md_up[i-1]==md_dif_up_3 and not bound_up_3_is_filled:
                    bound_up_3=md_up[i-1]
                    bound_up_3_is_filled=True

            arr_up=[bound_up_1,bound_up_2,bound_up_3]
            arr_up=sorted(arr_up)

            min_val=df[X_name].min()
            max_val=df[X_name].max()


          
            

            for j in range(1,len(md_down)):
                if md_down[j]-md_down[j-1]==md_dif_down_1 and not bound_down_1_is_filled:
                    bound_down_1=md_down[j-1]
                    bound_down_1_is_filled=True
                elif md_down[j]-md_down[j-1]==md_dif_down_2 and not bound_down_2_is_filled:
                    bound_down_2=md_down[j-1]
                    bound_down_2_is_filled=True

            arr_down=[bound_down_1,bound_down_2]
            arr_down=sorted(arr_down)

            

            NC_cut=st.slider(
            "NC",
            min_value=0.0,
            max_value=float(max_val),
            value=0.0,
            step=1.0
            )

           
            C_minus_cut = st.slider(
            "C-",
            min_value=float(NC_cut+1),
            max_value=float(max_val),
            value=float(avg - arr_down[1]-1),   
            step=1.0
            )
            C_cut = st.slider(
            "C",
            min_value=float(C_minus_cut+1),
            max_value=float(max_val),
            value=float(avg - arr_down[0]-1),   
            step=1.0
            )
            B_minus_cut = st.slider(
            "B-",
            min_value=float(C_cut+1),
            max_value=float(max_val),
            value=avg-1,   
            step=1.0
            )
            B_cut = st.slider(
            "B",
            min_value=float(B_minus_cut+1),
            max_value=float(max_val),
            value=float(avg + arr_up[0]-1),   
            step=1.0
            )
            A_minus_cut = st.slider(
            "A-",
            min_value=float(B_cut+1),
            max_value=float(max_val),
            value=float(avg + arr_up[1]-1),   
            step=1.0
            )
            A_cut = st.slider(
            "A",
            min_value=float(A_minus_cut+1),
            max_value=float(max_val),
            value=float(avg + arr_up[2]-1),   
            step=1.0
            )
            A_count=0
            A_minus_count=0
            B_count=0
            B_minus_count=0
            C_count=0
            C_minus_count=0
            NC_count=0

            for num in df[X_name]:
                if num>A_cut:
                    A_count+=1
                elif num>A_minus_cut:
                    A_minus_count+=1
                elif num>B_cut:
                    B_count+=1
                elif num>B_minus_cut:
                    B_minus_count+=1
                elif num>C_cut:
                    C_count+=1
                elif num>C_minus_cut:
                    C_minus_count+=1
                else:
                    NC_count+=1


            if st.button("Generate Histogram plot"):
            
             plt.clf()
             plt.title(title_inp)
             avg_val=round(df[X_name].mean(),2)
             plt.xlabel(X_label)
             plt.ylabel(Y_label)

             if is_bound=="Yes":
                plt.figure(figsize=(16,8))
                plt.axvline(A_cut,color='red')
                plt.axvline(A_minus_cut, color='red')
                plt.axvline(B_cut, color='red')
                plt.axvline(avg, color='blue')
                plt.axvline(C_cut, color='red')
                plt.axvline(C_minus_cut, color='red')
                plt.axvline(NC_cut, color="black")

                st.write(f"A cutoff={A_cut};  A : {A_count}")
                st.write(f"A minus cutoff={A_minus_cut};  A- : {A_minus_count}")
                st.write(f"B cutoff={B_cut};  B : {B_count}")
                st.write(f"B- cutoff={round(avg, 2)};  B- : {B_minus_count}")
                st.write(f"C cutoff={C_cut};  C : {C_count}")
                st.write(f"C- cutoff={C_minus_cut};  C- : {C_minus_count}")
                st.write(f"NC cutoff={NC_cut}; NC : {NC_count}")


                
            
            
            
            





            # Draw histogram AND get counts + bins
            
            plt.xticks(range(int(min_val), int(max_val) + 1, 3), fontsize=15)
            plt.hist(df[X_name],bins=num_bins,edgecolor='black')
            st.write("Note- ensure that a 'Grade' column doesnt exist in your spreadsheet")
            
            df["Grade"] = df[X_name].apply(
               lambda x: (
               'A'  if x > A_cut else
               'A-' if x > A_minus_cut else
               'B'  if x > B_cut else
               'B-' if x > B_minus_cut else
               'C'  if x > C_cut else
               'C-' if x > C_minus_cut else
               'NC'
               )
            )


    # SHOW UPDATED DF
            st.dataframe(df)

    # DOWNLOAD UPDATED CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
            label="Download Updated CSV",
            data=csv,
            file_name="updated_file.csv",
            mime="text/csv"
            )
            sum_grade=0
            for ch in df["Grade"]:
                if ch=="A":
                    sum_grade+=10
                elif ch=="A-":
                    sum_grade+=9
                elif ch=="B":
                    sum_grade+=8
                elif ch=="B-":
                    sum_grade+=7
                elif ch=="C":
                    sum_grade+=6
                elif ch=="C-":
                    sum_grade+=5
                elif ch=="NC":
                    sum_grade+=0
        
            len_grade=0
            for grade in df["Grade"]:
                if grade!="NC":
                    len_grade+=1

            MGPV=round(sum_grade/len_grade, 2)
            Average=round(df[X_name].mean(),2)
            
            st.text(f"Average= {Average:.2f} | MGPV= {MGPV:.2f} | Course Code= {Course_code} | Course name: {Course_name}")

            plt.xticks(range(int(min_val), int(max_val) + 1, 3), fontsize=10)



            st.pyplot(plt)
        
        elif is_bound=="No" and st.button("Generate Histogram plot"):
             plt.clf()
             plt.figure(figsize=(16,8))
             plt.title(title_inp)
             plt.xlabel(X_label)
             plt.ylabel(Y_label)
             plt.hist(df[X_name], bins=num_bins, edgecolor='black')
             plt.xticks(range(int(min_val), int(max_val) + 1, 3), fontsize=10)

             st.pyplot(plt)

        


