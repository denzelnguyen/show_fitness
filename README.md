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
| `gyr_x/y/z (deg/s)`| 3-axis gyroscope readings (degrees per second)          |

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
## 3. Related Work

This project is inspired by and built upon several influential studies in the fields of **Human Activity Recognition (HAR)**, **fitness analytics**, and **wearable sensor–based machine learning**. The following works provide important methodological and empirical foundations.

---

### 3.1 Intelligent Fitness Data Analysis and Training Effect Prediction  
**Paper:** *Intelligent Fitness Data Analysis and Training Effect Prediction Based on Machine Learning Algorithms*

**Models & Approach**
- Evaluates multiple supervised learning models for activity classification:
  - **Random Forest (RF):** Main model, using ensemble decision trees to improve robustness and reduce overfitting.
  - **Support Vector Machine (SVM):** Used to construct optimal decision boundaries between activities.
  - **Naive Bayes (NB):** Serves as a lightweight baseline for speed comparison.
- Workflow follows a classical pipeline of **signal preprocessing, feature extraction, and classification**, similar to this project.

**Dataset**
- Collected from wearable devices (smartwatches / IMU sensors).
- Time-series data including **3-axis accelerometer** and **heart rate signals**.
- Activities include running, cycling, squats, and push-ups.

**Key Findings**
- Random Forest achieved the highest accuracy (>90%).
- RF demonstrated strong robustness under noisy sensor conditions.
- The study confirms the feasibility of predicting training effectiveness using ML-based activity recognition.

---

### 3.2 Activity Recognition using Smartphone and Smartwatch Sensors (WISDM)  
**Authors:** Kwapisz et al.  
**Source:** ACM SIGKDD Explorations / IEEE Access / UCI WISDM Dataset

**Models & Approach**
- Proposes a structured HAR pipeline:
  - Fixed-length **time-window segmentation**
  - Noise reduction and outlier handling
  - Extraction of over 40 statistical and domain-specific features
  - **PCA** for dimensionality reduction
- Classification using:
  - Naive Bayes
  - Random Forest
  - Support Vector Machine

**Dataset**
- **WISDM Dataset**
- 51 participants, 18 activities
- Dual-device setup:
  - Smartphone (pocket)
  - Smartwatch (dominant hand)
- Sensors: tri-axial accelerometer and gyroscope

**Key Findings**
- Random Forest achieved the highest overall accuracy (>90%).
- Smartwatch data is more effective for upper-body activities.
- Smartphone data performs better for locomotion tasks.
- Sensor fusion significantly improves recognition performance.

---

### 3.3 A User-Adaptive Algorithm for Activity Recognition and Repetition Counting

**Models & Approach**
- Introduces a **user-adaptive HAR framework**:
  - **K-Means clustering** for initializing user-specific activity patterns
  - **Local Outlier Factor (LOF)** for removing unreliable samples
  - **Multivariate Gaussian models** for activity intensity classification
- Includes probability-based fall detection.

**Dataset**
- IMU sensor data (accelerometer-based)
- Activities grouped by intensity: light, moderate, vigorous
- Multi-user dataset for adaptability evaluation

**Key Findings**
- User-adaptive learning improves recognition accuracy.
- Reduces manual labeling requirements.
- Suitable for real-time wearable and healthcare applications due to low computational cost.

---

### 3.4 Physical Activity Monitoring using the PAMAP2 Dataset

**Models & Approach**
- Preprocessing: interpolation, noise filtering, outlier removal
- Feature extraction from both time and frequency domains (FFT, signal energy)
- **PCA** applied for dimensionality reduction
- Classification using Naive Bayes, SVM, and Random Forest

**Dataset**
- **PAMAP2 Dataset (UCI)**
- 9 participants
- 3 IMUs (wrist, chest, ankle) + heart rate monitor
- 18 activities, 52 attributes per timestamp

**Key Findings**
- Random Forest achieved the best performance (>90% accuracy).
- Naive Bayes was computationally efficient but less accurate.
- Multi-sensor fusion significantly improved recognition accuracy.
- PCA enabled near real-time inference.

---

### 3.5 RecGym: Gym Workouts Recognition Dataset  
**Authors:** Ceriani et al.  
**Source:** Sensors (MDPI), 2020

**Models & Approach**
- Signal smoothing using low-pass and moving average filters
- Feature extraction:
  - Signal Magnitude Area (SMA)
  - Peak-to-peak values
  - Time-domain statistics
- PCA for dimensionality reduction
- Classification using Random Forest (primary) and SVM
- Repetition counting via peak detection and zero-crossing analysis

**Dataset**
- **RecGym Dataset**
- 10 participants
- Exercises: Squat, Bench Press, Arm Curl, Leg Curl, etc.
- Sensors: IMU + capacitive sensors

**Key Findings**
- Random Forest achieved F1-score >94%
- Repetition counting error <3%
- Multimodal sensor fusion outperformed IMU-only approaches

---

