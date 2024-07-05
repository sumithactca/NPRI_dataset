import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Load the dataset (assuming 'df_original_price.csv' is your dataset)
df = pd.read_csv("df_original_price.csv")

# Set page title and favicon
st.set_page_config(page_title="Predicted Quantities", page_icon="📈")

# Header
st.title("Predicted Quantities from 2017 to 2025")

# Sidebar for user input
st.sidebar.header('User Input Features')
Company_name = st.sidebar.selectbox('Company name', df['Company name'].unique())
Substance_name = st.sidebar.selectbox('Substance name', df['Substance name'].unique())
Number_of_employees = st.sidebar.number_input('Number of employees', min_value=1, max_value=5000, value=10, step=1)
Province = st.sidebar.selectbox('Province', df['Province'].unique())
price = st.sidebar.number_input('Price')

# Preprocess the data (excluding 'Current year quantity')
X = df[['Company name', 'Substance name', 'Number of employees', 'Province', 'price',
        'Last year quantity', 'Two years ago quantity', 'Three years ago quantity']]
y = df['Current year quantity']

# Function to predict quantities for future years
def predict_quantities(company_name, substance_name, number_of_employees, province, price):
    # Filter data based on user inputs (assuming single company and substance)
    filtered_data = df[(df['Company name'] == company_name) & (df['Substance name'] == substance_name)]

    # Train a model (using data up to 2022)
    X_train = filtered_data[['Last year quantity', 'Two years ago quantity', 'Three years ago quantity',
                             'Number of employees', 'price']][filtered_data['NPRI_Report_ReportYear'] < 2023]
    y_train = filtered_data['Current year quantity'][filtered_data['NPRI_Report_ReportYear'] < 2023]

    # Initialize predictions list
    predictions = []

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict for 2017 to 2025
    years = list(range(2017, 2026))
    predicted_quantities = []

    for year in years:
        if year < 2023:
            # Fetch current year quantity from the original dataset
            quantity = filtered_data['Current year quantity'][(filtered_data['NPRI_Report_ReportYear'] == year)]
            predicted_quantities.append(quantity.iloc[0] if not quantity.empty else None)
        else:
            # Predict quantity using the machine learning model
            input_data = [[filtered_data['Last year quantity'].iloc[-1],  # Assuming last available year
                           filtered_data['Two years ago quantity'].iloc[-1],
                           filtered_data['Three years ago quantity'].iloc[-1],
                           number_of_employees,
                           price]]
            prediction = model.predict(input_data)[0]
            predicted_quantities.append(prediction)

            # Update input data for the next prediction
            filtered_data.loc[len(filtered_data)] = {
                'Last year quantity': filtered_data['Two years ago quantity'].iloc[-1],
                'Two years ago quantity': filtered_data['Three years ago quantity'].iloc[-1],
                'Three years ago quantity': prediction,
                'Number of employees': number_of_employees,
                'price': price,
                'Company name': company_name,
                'Substance name': substance_name,
                'Province': province,
                'NPRI_Report_ReportYear': year,
            }

    return years, predicted_quantities

# Get predictions
years, predicted_quantities = predict_quantities(Company_name, Substance_name, Number_of_employees, Province, price)

# Plotting the predicted quantities using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=years, y=predicted_quantities,
                         mode='lines+markers',
                         name='Predicted Quantities',
                         line=dict(color='blue', width=2),
                         marker=dict(color='blue', size=8, line=dict(color='white', width=2))))

# Adding titles and labels
fig.update_layout(title='Predicted Quantities from 2017 to 2025',
                  xaxis_title='Year',
                  yaxis_title='Quantity',
                  template='plotly_white',
                  font=dict(family="Arial, sans-serif", size=12, color="black"),
                  margin=dict(l=50, r=50, t=50, b=50),
                  xaxis=dict(tickmode='linear', tickvals=years, ticktext=years))  # Setting tick values and text

# Display the plot
st.plotly_chart(fig)
