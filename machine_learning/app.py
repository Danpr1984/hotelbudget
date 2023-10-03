import streamlit as st
import pandas as pd 
import joblib
import os


#Load the occupancy model
occupancy_model = joblib.load('/workspace/hotelbudget/occupancy_model.pkl') 
fb_occ_model = joblib.load('/workspace/hotelbudget/f&b_occ_model.pkl')
fb_revenue_model = joblib.load('/workspace/hotelbudget/f&b_revenue_model.pkl')
op_expenses_model = joblib.load('/workspace/hotelbudget/expenses_model.pkl')
rooms_expenses_model = joblib.load('/workspace/hotelbudget/rooms_expenses_model.pkl')
fb_expenses_model = joblib.load('/workspace/hotelbudget/fb_expenses_model.pkl')


# #Load the models using environment variables
# occupancy_model_path = os.environ.get('OCCUPANCY_MODEL_PATH')
# fb_occ_model_path = os.environ.get('FB_OCC_MODEL_PATH')
# fb_revenue_model_path = os.environ.get('FB_REVENUE_MODEL_PATH')
# op_expenses_model_path = os.environ.get('OP_EXPENSES_MODEL_PATH')
# rooms_expenses_model_path = os.environ.get('ROOMS_EXPENSES_MODEL_PATH')
# fb_expenses_model_path = os.environ.get('FB_EXPENSES_MODEL_PATH')

# # Check if any of the model paths are None (not found in environment variables)
# if None in [occupancy_model_path, fb_occ_model_path, fb_revenue_model_path, op_expenses_model_path, rooms_expenses_model_path, fb_expenses_model_path]:
#     raise ValueError("One or more model paths are missing in environment variables.")

# # Load the models using the specified paths
# occupancy_model = joblib.load(occupancy_model_path)
# fb_occ_model = joblib.load(fb_occ_model_path)
# fb_revenue_model = joblib.load(fb_revenue_model_path)
# op_expenses_model = joblib.load(op_expenses_model_path)
# rooms_expenses_model = joblib.load(rooms_expenses_model_path)
# fb_expenses_model = joblib.load(fb_expenses_model_path)

# Use the models as needed in your app

# Set other model paths as needed

# Initialize session state variables for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Quick Summary"

# Create session state variable for selected quarter
if 'selected_quarter' not in st.session_state:
    st.session_state.selected_quarter = None

if 'total_revenue' not in st.session_state:
    st.session_state.total_revenue = 0  # Initialize total_revenue to 0 or any default value you prefer
if 'room_revenue' not in st.session_state:
    st.session_state.room_revenue = 0   
if 'fb_revenue' not in st.session_state:
    st.session_state.fb_revenue = 0    
if 'marketing_value' not in st.session_state:
    st.session_state.marketing_value = 0
if 'number_of_days' not in st.session_state:
    st.session_state.number_of_days = 0

# Create session state variables to track calculated values and selected month
if 'calculated_values' not in st.session_state:
    st.session_state.calculated_values = {
        "Total Revenue": 0,
        "Rooms Revenue": 0,
        "F&B Revenue": 0,
        "Operations Expenses": 0,
        "Rooms Expenses": 0,
        "F&B Expenses": 0,
    }

if 'selected_month' not in st.session_state:
    st.session_state.selected_month = "January"


# Define Pages
# Define Quick Summary Page
def quick_summary_page():
    st.title("Quick Summary")
    
    # Introduction to the ML Model
    st.header("Machine Learning Model Overview")
    st.write("Our machine learning model is designed to predict key indicators for hotel management based on real-world data. This application is now working with real data from a 9 bedroom boutique hotel located in the pacific coast of Nicaragua. The models could be adapted to other hotel businesses if they have enought data.")
    
    # Model Benefits
    st.header("Benefits of the ML Model")
    st.write("This ML model offers several advantages for hotel management:")
    
    # List of Benefits
    st.markdown("1. **Accurate Predictions**: The model provides accurate predictions for occupancy, revenue, expenses, and more, allowing for better decision-making.")
    st.markdown("2. **Cost Efficiency**: By predicting key financial indicators, the model helps in optimizing costs and maximizing revenue.")
    st.markdown("3. **Real-time Insights**: With real-time data inputs, the model offers up-to-date insights, crucial for dynamic hotel operations.")
    st.markdown("4. **Improved Planning**: Hotel managers can use these predictions for better planning, such as staffing, pricing, and resource allocation.")
    
    # Real-Life Applications
    st.header("Real-Life Applications")
    st.write("Our ML model can be applied in various real-life hotel management scenarios:")
    
    # List of Applications
    st.markdown("- **Pricing Strategy**: Determine optimal room rates based on predicted occupancy and demand.")
    st.markdown("- **Staffing Optimization**: Plan staffing levels efficiently based on expected occupancy.")
    st.markdown("- **Revenue Maximization**: Identify opportunities to increase revenue through targeted marketing or pricing adjustments.")
    
    # Conclusion
    st.write("Our machine learning model is a powerful tool for hotel management, offering accurate predictions and real-time insights. It empowers hotel managers to make data-driven decisions, optimize costs, and maximize revenue.")


def methodology_and_analysis_page():
    st.title("Methodology and Analysis")

    st.write("For this model, we employed regression algorithms due to their effectiveness in predicting continuous numeric values. Regression techniques were chosen to estimate key financial indicators such as revenue, expenses, and occupancy percentages which ultimately will provide a Gross Operating Profit result.")

            
    st.subheader("Pipelines")

    st.write("To enhance model performance, rigorous data cleaning and preprocessing were performed to ensure high data quality. Feature engineering techniques were employed to extract meaningful insights from raw data. Hyperparameter optimization fine-tuned model parameters, while cross-validation with gradient boosting ensured robustness and minimized overfitting.")
    
    st.subheader("**Occupancy % Model:**")
    st.code("""
def pipeline_random_forest_reg():
    pipeline = Pipeline([
        ("feature_scaling", StandardScaler()),
        ("model", ExtraTreesRegressor(random_state=101)),
    ])        return pipeline
            """)
    st.write("**Model Evaluation**")
    
    st.write("**Train Set**")
    st.write(f"R2 Score: 1.00 ")
    st.write(f"Mean Absolute Error: 0.0")
    st.write(f"Mean Squared Error: 0.0 ")
    st.write(f"Root Mean Squared Error: 0.0")
    
    st.write("**Test Set**")
    st.write(f"R2 Score: 0.944 ")
    st.write(f"Mean Absolute Error: 0.039")
    st.write(f"Mean Squared Error: 0.003")
    st.write(f"Root Mean Squared Error: 0.056")

    st.write("**Features the model was trained on for Occupancy %**")
    st.text("['Marketing', 'Seasonality', 'Average Room Rate', 'Local Rainy Season', 'Holidays Local']")

    st.subheader( "**F&B Revenue Model:**")
    st.code("""
def pipeline_random_forest_reg():
    pipeline = Pipeline([
        ("feature_scaling", StandardScaler()),
        ("model", RandomForestRegressor(random_state=101)),
    ])
    return pipeline
        """)
    st.write("**Model Evaluation**")

    st.write("**Train Set**")
    st.write(f"R2 Score: 0.964 ")
    st.write(f"Mean Absolute Error: 553.451")
    st.write(f"Mean Squared Error: 588924.729")
    st.write(f"Root Mean Squared Error: 767.414")

    st.write("**Test Set**")
    st.write(f"R2 Score: 0.828")
    st.write(f"Mean Absolute Error: 1219.316")
    st.write(f"Mean Squared Error: 2366472.62")
    st.write(f"Root Mean Squared Error: 767.414")

    st.write("**Features the model was trained on for F&B Revenue**")
    st.text("['Rooms Revenue', 'Seasonality', 'Holidays Local', 'Percentage Rooms Occ %', 'Percentage F&B Occ %']")

    st.subheader("**Expenses Model:**")
    st.code("""
def pipeline_random_forest_reg():
    pipeline = Pipeline([
        ("feature_scaling", StandardScaler()),
        ("model", RandomForestRegressor(random_state=101)),
    ])
    return pipeline
        """)
    st.write("**Model Evaluation**")

    st.write("**Train Set**")
    st.write(f"R2 Score: 0.976")
    st.write(f"Mean Absolute Error: 217.977")
    st.write(f"Mean Squared Error: 93848.892")
    st.write(f"Root Mean Squared Error: 306.348")

    st.write("**Test Set**")
    st.write(f"R2 Score: 0.915")
    st.write(f"Mean Absolute Error: 456.379")
    st.write(f"Mean Squared Error: 314733.454")
    st.write(f"Root Mean Squared Error: 561.011")

    st.write("**Features the model was trained on for Expenses**")
    st.text("['Total Revenue', 'Total Wages', 'Insuraces', 'Transport', 'Marketing', 'Maintenance', 'Utilities Expenses', 'Systems & Communications']")

    # Data Sources
    st.subheader("Data Source:")
    st.write("We separated the data from the main source, a hotel's monthly P&L's of 48 months to create models separately, focusing on the variables needed for each model.")
    # Real Hotel Data
    st.subheader("Revenue")
    st.write("Our analysis begins with real hotel data sourced from the file 'budgetusd.xlsx'.")
    
    # # Load the Excel file into a pandas DataFrame
    # excel_file_path = 'budgetusd.xlsx'
    # sheet_name = "Rooms Revenue"
    # df_real_hotel_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    

    # Load the Excel file into a pandas DataFrame
    excel_file_path = '/workspace/hotelbudget/budgetusd.xlsx'
    sheet_name = "Rooms Revenue"

    try:
        df_real_hotel_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # Display the first few rows of the DataFrame
            # Display the real hotel data table
        
        st.write("*** To augment our dataset and have more data values, we created synthetic data, not considering the Covid-19 years since this would affect drastically a prediction")
        st.subheader("Table: Real Rooms Revenue Hotel Data")
        st.dataframe(df_real_hotel_data.head(24))  # Display the first few rows of the real hotel data
    except FileNotFoundError:
        st.error("Authorization is required to see data values. Continue to ML Revenue Page to make predictions")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.write("Once we have predicted our Occupancy, we calculate Rooms Revenue by multiplying these values: predicted_occupancy * number_of_rooms * number_of_days * room_rate ")        
    st.write("")

        # Load the Excel file into a pandas DataFrame
    excel_file_path = '/workspace/hotelbudget/budgetusd.xlsx'
    sheet_name = "F&B Revenue"

    try:
        df_fb_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # Display F&B Revenue table
        st.subheader("Table: Real F&B Revenue Hotel Data")
        st.dataframe(df_fb_data.head(24))  # Display the first few rows of the real hotel data
    except FileNotFoundError:
        st.error("Authorization is required to see data values. Continue to ML Revenue Page to make predictions")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.subheader("Expenses")
    st.write("We separate the expenses by Rooms, Food and Beverage and othe operations expenses")

    # Load the Excel file into a pandas DataFrame
    excel_file_path = '/workspace/hotelbudget/budgetusd.xlsx'
    sheet_name = "Expenses"

    try:
        df_expenses_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # Display the first few rows of the DataFrame
            # Display the real hotel data table
                
        st.subheader("Table: Real Expenses Hotel Data")
        st.dataframe(df_expenses_data.head(24))  # Display the first few rows of the real hotel data
    except FileNotFoundError:
        st.error("Authorization is required to see data values. Continue to ML Revenue Page to make predictions")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Load the Excel file into a pandas DataFrame
    excel_file_path = '/workspace/hotelbudget/budgetusd.xlsx'
    sheet_name = "Rooms Expenses"

    try:
        df_rooms_expenses_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # Display the first few rows of the DataFrame
            # Display the real hotel data table
                
        st.subheader("Table: Rooms Expenses Hotel Data")
        st.dataframe(df_rooms_expenses_data.head(24))  # Display the first few rows of the real hotel data
    except FileNotFoundError:
        st.error("Authorization is required to see data values. Continue to ML Revenue Page to make predictions")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Load the Excel file into a pandas DataFrame
    excel_file_path = '/workspace/hotelbudget/budgetusd.xlsx'
    sheet_name = "F&B Expenses"

    try:
        df_fb_expenses_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # Display the first few rows of the DataFrame
            # Display the real hotel data table
                
        st.subheader("Table: F&B Expenses Hotel Data")
        st.dataframe(df_fb_expenses_data.head(24))  # Display the first few rows of the real hotel data
    except FileNotFoundError:
        st.error("Authorization is required to see data values. Continue to ML Revenue Page to make predictions")
    except Exception as e:
        st.error(f"An error occurred: {e}")    

        # Load the Excel file into a pandas DataFrame
    excel_file_path = '/workspace/hotelbudget/budgetusd.xlsx'
    sheet_name = "Indirect_Expenses"

    try:
        df_ops_expenses_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # Display the first few rows of the DataFrame
            # Display the real hotel data table
                
        st.subheader("Table: Operations Expenses Hotel Data")
        st.dataframe(df_ops_expenses_data.head(24))  # Display the first few rows of the real hotel data
    except FileNotFoundError:
        st.error("Authorization is required to see data values. Continue to ML Revenue Page to make predictions")
    except Exception as e:
        st.error(f"An error occurred: {e}")           

        
# Define a function to predict occupancy and room revenue
def predict_occupancy_and_revenue(marketing, seasonality, average_room_rate, rainy_season, holidays_local, number_of_rooms, number_of_days, room_rate):
    # Calculate occupancy percentage
    input_data_occupancy = {
        'Marketing': marketing,
        'Seasonality': seasonality,
        'Average Room Rate': average_room_rate,
        'Local Rainy Season': rainy_season,
        'Holidays Local': holidays_local,
    }
    predicted_occupancy = occupancy_model.predict([list(input_data_occupancy.values())])[0]

    # Calculate room revenue
    room_revenue = predicted_occupancy * number_of_rooms * number_of_days * room_rate

    return predicted_occupancy, room_revenue

# Define a function to predict F&B occupancy
def predict_fb_occupancy(room_revenue, holidays_local, rooms_occupancy, seasonality, rainy_season ):
    input_data_fb_occ = {
        'Rooms Revenue': room_revenue,
        'Holidays Local': holidays_local,
        'Percentage Rooms Occ %': rooms_occupancy,
        'Seasonality': seasonality,
        'Local Rainy Season': rainy_season,
        
    }
    fb_occupancy = fb_occ_model.predict([list(input_data_fb_occ.values())])[0]
    return fb_occupancy

# Define a function to predict F&B revenue
def predict_fb_revenue(room_revenue, seasonality, holidays_local, rooms_occupancy, fb_occupancy):
    input_data_fb_revenue = {
        'Rooms Revenue': room_revenue,
        'Seasonality': seasonality,
        'Holidays Local': holidays_local,
        'Percentage Rooms Occ %': rooms_occupancy,
        'Percentage F&B Occ %': fb_occupancy,
    }
    fb_revenue = fb_revenue_model.predict([list(input_data_fb_revenue.values())])[0]
    return fb_revenue


# Define ML Revenue Page
def ml_revenue_page():
    # Input widgets for occupancy prediction
    st.subheader("Occupancy Percentage & Revenue Prediction")

    selected_month = st.selectbox("Select a Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])


    #Season
    seasonality_label = st.selectbox("Select the Season", ("Low", "Medium", "High"))
    seasonality_value = {"Low": 0, "Medium": 1, "High": 2}[seasonality_label]

    # Marketing
    marketing_value = st.slider("Marketing Investment USD$ (0-500)", 0, 500, step=50)

    # Average Room Rate
    average_room_rate_value = st.number_input("Average Room Rate USD$", min_value=50, max_value=120, value=100)

    
    # Local Rainy Season
    rainy_season_label = st.selectbox("Select the Local Rainy Season", list({'No Rain': 0, 'Moderate': 1, 'Heavy Rain': 2}.keys()))
    rainy_season_mapping = {'No Rain': 0, 'Moderate': 1, 'Heavy Rain': 2}
    rainy_season = rainy_season_mapping[rainy_season_label]

    #Rooms
    number_of_rooms = st.number_input("Number of Rooms", value=9)
    number_of_days = st.number_input("Number of Days", value=30)
    room_rate = average_room_rate_value

    # Define the rainy_season variable here
    rainy_season = rainy_season_mapping[rainy_season_label]

    holidays_local = st.slider("Local Holidays", 0, 10, step=1)

    if st.button("Calculate Occupancy and Revenue"):
    
    # Calculate occupancy and room revenue using the custom function
        predicted_occupancy, room_revenue = predict_occupancy_and_revenue(
            marketing_value,
            seasonality_value,
            average_room_rate_value,
            rainy_season,
            holidays_local,
            number_of_rooms,
            number_of_days,
            room_rate,
        )
    
        st.write(f"**The predicted occupancy percentage is: {predicted_occupancy * 100:.2f}%**")
        st.write(f"**The predicted room revenue is: ${room_revenue:.2f}**")

# Calculate F&B Occupancy using the custom function
        predicted_fb_occupancy = predict_fb_occupancy(
            room_revenue,  # Reuse rooms revenue from room revenue prediction
            predicted_occupancy,  # Reuse predicted occupancy from room occupancy prediction
            seasonality_value,  # Use the selected seasonality from the ML Revenue page
            holidays_local,  
            rainy_season,
        )

    #st.subheader("Predicted F&B Occupancy %")
    #st.write(f"The predicted F&B occupancy percentage is: {predicted_fb_occupancy * 100:.2f}%")

# Calculate F&B revenue using the custom function
        fb_revenue = predict_fb_revenue(
            room_revenue, 
            seasonality_value, 
            holidays_local, 
            predicted_occupancy, 
            predicted_fb_occupancy)

        st.write(f"**The predicted F&B revenue is: ${fb_revenue:.2f}**")

    # Calculate Total Revenue
        total_revenue = room_revenue + fb_revenue

        st.session_state.room_revenue = room_revenue
        st.session_state.total_revenue = total_revenue
        st.session_state.fb_revenue = fb_revenue
        st.session_state.marketing_value = marketing_value
        st.session_state.number_of_days = number_of_days

# Display Total Revenue
        st.subheader("Total Revenue")
        st.write(f"**The total revenue is: ${total_revenue:.2f}**")
        st.write("****Please note that the results are valid using data from a specific business so results are valid for that specific business. Data from other businesses could also be fed into the models.**")

        # Update the st.session_state.calculated_values dictionary
        if selected_month not in st.session_state.calculated_values:
            st.session_state.calculated_values[selected_month] = {}

            # Update the st.session_state.calculated_values dictionary
        st.session_state.calculated_values[selected_month]["Total Revenue"] = total_revenue
        st.session_state.calculated_values[selected_month]["Rooms Revenue"] = room_revenue
        st.session_state.calculated_values[selected_month]["F&B Revenue"] = fb_revenue


#Expenses Models


def predict_ops_expenses(total_revenue, total_wages, insurances, transport, marketing, maintenance, utilities, systems_communications):
    input_data_ops_exp = {
    'Total Revenue': total_revenue,
    'Total Wages': total_wages,
    'Insuraces': insurances,  
    'Transport': transport,
    'Marketing': marketing,
    'Maintenance': maintenance,
    'Utilities Expenses':utilities, 
    'Systems & Communications':systems_communications,
    }

    ops_expenses = op_expenses_model.predict([list(input_data_ops_exp.values())])[0]
    return ops_expenses


# Define a function to predict room expenses
def predict_rooms_expenses(room_revenue):
    input_data_rooms_expenses = {
        'Rooms Revenue': room_revenue
    }
    rooms_expenses = rooms_expenses_model.predict([list(input_data_rooms_expenses.values())])[0]
    return rooms_expenses

# Define a function to predict fb expenses
def predict_fb_expenses(fb_revenue):
    input_data_fb_expenses = {
        'F&B Revenue': fb_revenue
    }
    fb_expenses = fb_expenses_model.predict([list(input_data_fb_expenses.values())])[0]
    return fb_expenses

def ml_expenses_and_gop_page():
    st.title("ML Expenses and GOP")
    # Add content for the ML Expenses and GOP page here

    selected_month = st.selectbox("Select a Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    # Manual input for Total Revenue (from the previous page)
    total_revenue = st.session_state.total_revenue
    room_revenue = st.session_state.room_revenue
    fb_revenue = st.session_state.fb_revenue 
    marketing_value = st.session_state.marketing_value
    number_of_days = st.session_state.number_of_days

    # Display Total Revenue
    st.write(f"The total revenue is: **${total_revenue:.2f}**")

    # Display Rooms Revenue
    st.write(f"The rooms revenue is: **${room_revenue:.2f}**")

    # Display Rooms Revenue
    st.write(f"The F&B revenue is: **${fb_revenue:.2f}**")

    # Input widgets for manual input
    st.write(f"Fill expenses projection for **{number_of_days} days**")

    # Manual input for Marketing Investment
    marketing = st.number_input("Marketing Investment", marketing_value)

    # Manual input for Total Wages
    total_wages = st.number_input("Total Wages", min_value=0)
    
    # Manual input for Insurances
    insurances = st.number_input("Insurances", min_value=0)
    
    # Manual input for Transport
    transport = st.number_input("Transport", min_value=0)

    # Manual input for Maintenance
    maintenance = st.number_input("Maintenance", min_value=0)

    # Manual input for Utilities Expenses
    utilities = st.number_input("Utilities Expenses", min_value=0)

    systems_communications = st.number_input("Systems & Communications", min_value=0)
    

    
    # Calculate operational expenses using the predict_ops_expenses function
    if st.button("Calculate Operational Expenses"):
        ops_expenses = predict_ops_expenses(total_revenue, total_wages, insurances, transport, 
        marketing, maintenance, utilities, systems_communications)

        
        st.write(f"**The estimated operational expenses are: ${ops_expenses:.2f}**")

        # Calculate room expenses using the custom function
        room_expenses = predict_rooms_expenses(room_revenue)

    # Display Room Expenses
        st.write(f"**The predicted room expenses are: ${room_expenses:.2f}**")

    # Calculate f&b expenses using the custom function
        fb_expenses = predict_fb_expenses(fb_revenue)  

    # Display Room Expenses
        st.write(f"**The predicted f&b expenses are: ${fb_expenses:.2f}**")   

        total_expenses =  ops_expenses + room_expenses + fb_expenses  

        # Add "Total Expenses" to the calculated_values dictionary
        st.session_state.calculated_values[selected_month]["Total Expenses"] = total_expenses

        gross_operating_profit = total_revenue - total_expenses
        gop_margin = gross_operating_profit / total_revenue

        # Display Total Expenses
        st.subheader(f"**The total expenses are: ${total_expenses:.2f}**") 

        # Display Total Expenses
        st.subheader(f"**The total GOP is: ${gross_operating_profit:.2f} representing a margin of: {gop_margin * 100:.2f}%**")

        # Update the st.session_state.calculated_values dictionary
        if selected_month not in st.session_state.calculated_values:
            st.session_state.calculated_values[selected_month] = {}  
            # Update the st.session_state.calculated_values dictionary
        st.session_state.calculated_values[selected_month]["Operations Expenses"] = ops_expenses
        st.session_state.calculated_values[selected_month]["Rooms Expenses"] = room_expenses
        st.session_state.calculated_values[selected_month]["F&B Expenses"] = fb_expenses
        st.session_state.calculated_values[selected_month]["Total Expenses"] = total_expenses   


# Define Yearly Prediction Page
def yearly_prediction_page():
    st.title("Yearly Prediction")

    total_totals = {
        "Total Revenue": 0,
        "Rooms Revenue": 0,
        "F&B Revenue": 0,
        "Operations Expenses": 0,
        "Rooms Expenses": 0,
        "F&B Expenses": 0,
        "Total Expenses": 0
    }

    all_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    quarters = [
        "Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4"
    ]

    # Create tables for each row of months
    num_cols = 3

    for i in range(0, len(all_months), num_cols):
        month_slice = all_months[i:i + num_cols]
        st.write(" | ".join(month_slice))
        data = []


        
        for key in ["Total Revenue", "Rooms Revenue", "F&B Revenue", "Operations Expenses", "Rooms Expenses", "F&B Expenses"]:
            values = []
            for month in month_slice:
                if month in st.session_state.calculated_values:
                    value = st.session_state.calculated_values[month].get(key, 0)
                    values.append(f"${value:.2f}")
                else:
                    values.append("$0.00")
            data.append(values)

        # Create a DataFrame for the table
        df = pd.DataFrame(data, columns=month_slice, index=["Total Revenue", "Rooms Revenue", "F&B Revenue", "Operations Expenses", "Rooms Expenses", "F&B Expenses"])
        st.dataframe(df, use_container_width=True)

        # Add a separator between tables
        st.write("---")

    # Calculate and display the totals table
    st.header("Monthly Totals")
    
    for month in all_months:
        if month in st.session_state.calculated_values:
            month_data = st.session_state.calculated_values[month]
            for key, value in month_data.items():
                total_totals[key] += value

    # Create a DataFrame for the totals table
    totals_df = pd.DataFrame.from_dict(total_totals, orient='index', columns=['Total'])
    st.write("Totals")
    st.dataframe(totals_df, use_container_width=True)


def conclusion_page():
    st.title("Conclusion")
    # Add content for the Conclusion page here
    
    
    st.subheader("Elevating Hotel Financial Strategy")
    st.write("Our journey through data analysis and model development culminates in a resounding affirmation of our predictive models' effectiveness. With exceptional R2, MAE, and MSAE scores, we have harnessed the power of data to bolster budget forecasting for hotels.")
    st.write("These models are not just theoretical constructs but practical tools for the hotel industry. By accurately forecasting revenue, occupancy rates, and expenses, they empower hoteliers to make strategic decisions with confidence. Whether optimizing pricing strategies or efficiently allocating resources, our models provide a competitive edge in an ever-evolving market.")
    st.write("In an industry where adaptability is key, these models offer a path to resilience and prosperity. By integrating data-driven insights into their financial strategies, hotels can navigate uncertainties, enhance profitability, and secure a promising future.")
    
    

# Create a Sidebar Menu
selected_page = st.sidebar.radio("Navigation", ["Quick Summary", "Methodology and Analysis", "ML Revenue", "ML Expenses and GOP", "Yearly Prediction", "Conclusion"])

# Update current page using query parameters
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.experimental_set_query_params(page=selected_page)  # Add this line to update the URL

# Display Selected Page Content
if selected_page == "Quick Summary":
    quick_summary_page()
elif selected_page == "Methodology and Analysis":
    methodology_and_analysis_page()
elif selected_page == "ML Revenue":
    ml_revenue_page()
elif selected_page == "ML Expenses and GOP":
    ml_expenses_and_gop_page()
elif st.session_state.current_page == "Yearly Prediction":
    yearly_prediction_page()
elif st.session_state.current_page == "Conclusion":
    conclusion_page()