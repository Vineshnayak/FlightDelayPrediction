# Importing necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Loading the dataset
data = pd.read_csv('flight_delay_predict.csv')

# Displaying the first few rows of the data
print("First few rows of the dataset:")
print(data.head())

# Data Preprocessing

# Convert 'FlightDate' to datetime
data['FlightDate'] = pd.to_datetime(data['FlightDate'])

# Visualizing Average Arrival Delay by Origin Airport
avg_delay_by_origin = data.groupby('Origin')['ArrDelay'].mean().reset_index()
bar_plot = px.bar(avg_delay_by_origin, x='Origin', y='ArrDelay', title='Average Arrival Delay by Origin Airport')
bar_plot.update_layout(xaxis_title='Origin Airport', yaxis_title='Average Arrival Delay (minutes)')
bar_plot.show()

# Visualizing Average Arrival Delay by Destination Airport
avg_delay_by_dest = data.groupby('Dest')['ArrDelay'].mean().reset_index()
bar_plot_dest = px.bar(avg_delay_by_dest, x='Dest', y='ArrDelay', title='Average Arrival Delay by Destination Airport')
bar_plot_dest.update_layout(xaxis_title='Destination Airport', yaxis_title='Average Arrival Delay (minutes)')
bar_plot_dest.show()

# Correlation matrix of numeric features
numeric_data = data.select_dtypes(include=['number'])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# Average Delay by Month
avg_delay_month = data.groupby(data['FlightDate'].dt.month)['is_delay'].mean().reset_index()
fig = px.bar(avg_delay_month, x='FlightDate', y='is_delay', labels={'FlightDate': 'Month', 'is_delay': 'Average Delay'},
             title='Average Delay by Month')
fig.update_traces(marker_color='skyblue')
fig.show()

# Splitting the data into training and testing sets
X = data[['AirTime', 'Distance']]
y = data[['ArrDelayMinutes', 'is_delay']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the Neural Network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='linear'))  # Output layer for predicted delay time and delay status

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Training the model
print("Training the model...")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Evaluating the model
score, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Evaluation:\nScore (Loss): {score:.4f}\nAccuracy: {accuracy:.4f}")

# Saving the model
model.save('/kaggle/working/model.h5')

# Real-time Prediction
print("\nReal-time Prediction:")
try:
    air_time = float(input("Enter Air Time in minutes (e.g., 150): "))
    distance = float(input("Enter Distance in miles (e.g., 1000): "))
    
    # Preparing user input for prediction
    user_input = np.array([[air_time, distance]])
    user_input_scaled = scaler.transform(user_input)
    
    # Making predictions
    predictions = model.predict(user_input_scaled)

    # Interpreting the predictions
    predicted_delay = predictions[0][0]
    delay_status = predictions[0][1]

    # Displaying results in an understandable format
    if delay_status >= 0.5:
        print(f"\n🚨 Prediction: The flight is predicted to be delayed by approximately {predicted_delay:.2f} minutes.")
    else:
        print("\n✅ Prediction: The flight is predicted to NOT be delayed.")

except ValueError:
    print("\n⚠️ Please enter valid numerical values for Air Time and Distance.")