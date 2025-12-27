# Sports & Exercise Activity Recognition Using IMU Sensor Data

## 1. Introduction

This repository contains the final project for the course **Human Activity Recognition & Machine Learning**.  
The goal is to develop an automatic activity recognition system using **IMU motion sensor data** (accelerometer & gyroscope), capable of classifying and counting physical activities such as walking, running, squatting, and strength exercises.

This work simulates features found in smart wearable devices (e.g., smartwatches) for **automatic exercise detection** and **repetition counting**, supporting the concept of **Quantified Self** — leveraging personal data to better understand and improve physical activity habits.

---

## 2. Problem Statement

Human activity recognition (HAR) from wearable sensors plays an important role in health monitoring, sports analytics, and personalized fitness tracking.

The system analyzes time-series signals from 3-axis accelerometer and gyroscope sensors to automatically distinguish between different labeled activities using a trained machine learning model.

Key objectives:

1. Collect and preprocess sensor time-series data.
2. Train a classification model to recognize activity types.
3. Evaluate model performance on held-out test data.
4. Support activity counting for repetitive movement exercises.

---

## 3. Dataset

### 3.1 Data Source  
- **Primary Sensor:** MetaMotion Sensor  
- **Data Format:** CSV (Comma-Separated Values)

Each record includes synchronized sensor readings with:

| Field            | Description                                                |
|------------------|------------------------------------------------------------|
| `epoch (ms)`     | Timestamp in milliseconds for synchronization              |
| `time`           | Human-readable date and time of capture                    |
| `elapsed (s)`    | Duration (seconds) since the start of recording            |
| `acc_x/y/z (g)`  | 3-axis accelerometer readings (G-force units)              |
| `gyro_x/y/z (º/s)` | 3-axis gyroscope readings (degrees per second)          |

### 3.2 Activities & Participants  
- **Tracked Activities:**  
  - Bench press  
  - Deadlift  
  - Overhead press  
  - Squat  
  - Walking  
  - Running

- **Participants:**  
  - Data collected from **5 volunteers** following a structured strength training program.
