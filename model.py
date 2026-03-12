import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("D:\Food Delivery Prediction Model\csv\Food_Delivery_Times.csv")

# Drop Order_ID
df = df.drop("Order_ID", axis=1)

# Encode categorical columns
le_weather = LabelEncoder()
le_traffic = LabelEncoder()
le_time = LabelEncoder()
le_vehicle = LabelEncoder()

df["Weather"] = le_weather.fit_transform(df["Weather"])
df["Traffic_Level"] = le_traffic.fit_transform(df["Traffic_Level"])
df["Time_of_Day"] = le_time.fit_transform(df["Time_of_Day"])
df["Vehicle_Type"] = le_vehicle.fit_transform(df["Vehicle_Type"])

# Features and target
X = df.drop("Delivery_Time_min", axis=1)
y = df["Delivery_Time_min"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("delivery_model.pkl", "wb"))

print("Model trained and saved successfully!")