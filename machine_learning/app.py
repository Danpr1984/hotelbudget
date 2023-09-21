import streamlit as st
import pandas as pd 
import joblib

# Define Pages
# Define Quick Summary Page
def quick_summary_page():
    st.title("Quick Summary")
    
    # Introduction to the ML Model
    st.header("Machine Learning Model Overview")
    st.write("Our machine learning model is designed to predict key indicators for hotel management based on real-world data.")
    
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
    st.header("Conclusion")
    st.write("Our machine learning model is a powerful tool for hotel management, offering accurate predictions and real-time insights. It empowers hotel managers to make data-driven decisions, optimize costs, and maximize revenue.")


def methodology_and_analysis_page():
    st.title("Methodology and Analysis")
    
    # Data Sources
    st.header("Data Sources")
    
    # Real Hotel Data
    st.subheader("Real Hotel Data")
    st.write("Our analysis begins with real hotel data sourced from the file 'budgetusd.xlsx'.")
    
    # Load the Excel file into a pandas DataFrame
    excel_file_path = '../budgetusd.xlsx'
    sheet_name = "Rooms Revenue"
    df_real_hotel_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    # Display the real hotel data table
    st.write("Table: Real Hotel Data")
    st.dataframe(df_real_hotel_data)  # Display the first few rows of the real hotel data
    
    # Synthetic Data
    st.subheader("Synthetic Data")
    st.write("To augment our dataset and have more data values, we created synthetic data.")
    
    # Load and display the synthetic data table (adjust the path and sheet name accordingly)
    excel_file_path_synthetic = '../synthetic_data.xlsx'
    sheet_name_synthetic = "Synthetic Data"
    df_synthetic_data = pd.read_excel(excel_file_path_synthetic, sheet_name=sheet_name_synthetic)
    
    st.write("Table: Synthetic Data")
    st.dataframe(df_synthetic_data.head())  # Display the first few rows of the synthetic data
    
    # Data Preparation
    st.header("Data Preparation")
    
    # Data Separation
    st.subheader("Data Separation")
    st.write("We separated the data to create models separately, focusing on the variables needed for each model.")

# Load the occupancy model
occupancy_model = joblib.load('/workspace/hotelbudget/predictive_models/occupancy_model.pkl')
food_andbev_model = joblib.load('/workspace/hotelbudget/f&b_revenue_model.pkl')

# Define a function to predict occupancy and room revenue
def predict_occupancy_and_revenue(marketing, seasonality, average_room_rate, number_of_rooms, number_of_days, room_rate):
    # Calculate occupancy percentage
    input_data_occupancy = {
        'Marketing': marketing,
        'Seasonality': seasonality,
        'Average Room Rate': average_room_rate,
    }
    predicted_occupancy = occupancy_model.predict([list(input_data_occupancy.values())])[0]

    # Calculate room revenue
    room_revenue = predicted_occupancy * number_of_rooms * number_of_days * room_rate

    return predicted_occupancy, room_revenue
def predict_food_andbev(percentage_fb, room_revenue, predicted_occupancy )
# Define ML Revenue Page
def ml_revenue_page():
    st.title("ML Revenue Page")

    # Input widgets for occupancy prediction
    st.header("Occupancy % & Room Revenue Prediction")

    #Season
    seasonality_label = st.selectbox("Select the Season", ("Low", "Medium", "High"))
    seasonality_value = {"Low": 0, "Medium": 1, "High": 2}[seasonality_label]

    # Marketing
    marketing_value = st.slider("Marketing Investment USD$ (0-500)", 0, 500, step=50)

    # Average Room Rate
    average_room_rate_value = st.number_input("Average Room Rate USD$", min_value=50, max_value=120, value=100)

    number_of_rooms = st.number_input("Number of Rooms", value=9)
    number_of_days = st.number_input("Number of Days", value=30)
    room_rate = average_room_rate_value

    # Calculate occupancy and room revenue using the custom function
    predicted_occupancy, room_revenue = predict_occupancy_and_revenue(
        marketing_value,
        seasonality_value,
        average_room_rate_value,
        number_of_rooms,
        number_of_days,
        room_rate,
    )

    st.subheader("Predicted Occupancy %")
    st.write(f"The predicted occupancy percentage is: {predicted_occupancy * 100:.2f}%")

    st.subheader("Predicted Room Revenue")
    st.write(f"The predicted room revenue is: ${room_revenue:.2f}")

# Rest of your Streamlit app code...


def ml_expenses_and_gop_page():
    st.title("ML Expenses and GOP")
    # Add content for the ML Expenses and GOP page here

def conclusion_page():
    st.title("Conclusion")
    # Add content for the Conclusion page here

# Create a Sidebar Menu
selected_page = st.sidebar.radio("Navigation", ["Quick Summary", "Methodology and Analysis", "ML Revenue", "ML Expenses and GOP", "Conclusion"])

# Display Selected Page Content
if selected_page == "Quick Summary":
    quick_summary_page()
elif selected_page == "Methodology and Analysis":
    methodology_and_analysis_page()
elif selected_page == "ML Revenue":
    ml_revenue_page()
elif selected_page == "ML Expenses and GOP":
    ml_expenses_and_gop_page()
elif selected_page == "Conclusion":
    conclusion_page()
