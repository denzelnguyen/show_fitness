1. Problem Description
- This project aims to build a machine learning–based activity recognition system using motion sensor data collected from Inertial Measurement Units (IMU), specifically accelerometers and gyroscopes.
- The system analyzes time-series signals from 3-axis accelerometer and gyroscope sensors to automatically classify different sports and exercise activities, such as walking, running, squatting, and strength-training movements. A trained machine learning model is used to learn patterns in these signals and make activity predictions.
- The main objective is to simulate features commonly found in smart wearable devices (e.g., smartwatches), including automatic exercise detection and repetition counting. This project aligns with the concept of Quantified Self, where personal sensor data is leveraged to monitor, analyze, and improve physical activity and lifestyle habits.

2. Dataset Description
2.1 Data Source
- Dataset Name: MetaMotion Sensor Dataset
- Data Format: CSV 
- Each CSV file contains synchronized sensor readings recorded over time during physical activities.
2.2 Sensor Data Structure
- Each data file includes the following fields:
    + epoch (ms): Timestamp in milliseconds, used to precisely synchronize data from multiple sensors.
    + time: Human-readable date and time indicating when the data was recorded.
    + elapsed (s): Time in seconds since the start of the recording session.
    + Accelerometer (acc) – x, y, z (g): Measures linear acceleration along three axes using gravitational force units (g).
    + Gyroscope (gyro) – x, y, z (deg/s):
    + Measures angular velocity (rotational motion) along three axes in degrees per second.
2.3 Activities and Participants
- Tracked Activities:
    + Bench Press
    + Deadlift
    + Overhead Press
    + Squat
    + Walking
    + Running
- Participants: Data was collected from 5 volunteers following a structured and professional strength training program.



