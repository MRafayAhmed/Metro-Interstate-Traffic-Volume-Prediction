# Metro-Interstate-Traffic-Volume-Prediction


A machine learning project that analyzes and predicts interstate traffic volume using multiple supervised, unsupervised, and reinforcement learning techniques.

---

##  Project Overview

This project uses the **Metro Interstate Traffic Volume** dataset to explore traffic patterns and build predictive models. The goal is to classify traffic levels (Low / Medium / High), predict exact traffic volume, cluster similar traffic conditions, and optimize traffic signal timing using Q-Learning.

---

##  Project Structure

```
TestProject/
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv   # Dataset
├── Metrodata.py                               # Dataset downloader (via KaggleHub)
├── model.py                                   # Main ML pipeline
├── Metro-Interstate-Traffic-Volume-Prediction.pptx  # Presentation slides
└── Project report.pdf                         # Full project report
```

---

##  Dataset

- **Source:** [Kaggle – Metro Interstate Traffic Volume](https://www.kaggle.com/datasets/pooriamst/metro-interstate-traffic-volume)
- **Features:** `holiday`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`, `date_time`
- **Target:** `traffic_volume` (number of vehicles per hour)

---

##  Features Engineered

- `year`, `month`, `day`, `hour`, `dayofweek` — extracted from `date_time`
- `is_rush_hour` — 1 if hour is between 7–9 AM or 5–7 PM
- `is_weekend` — 1 if day is Saturday or Sunday
- `weather_severity` — mapped from weather condition (0=Clear to 3=Thunderstorm)

---

##  Models Implemented

### Classification (Traffic Level: Low / Medium / High)
| Model | Description |
|---|---|
| **K-Nearest Neighbors (KNN)** | k=5, classifies based on nearest training samples |
| **Support Vector Machine (SVM)** | RBF kernel |
| **Decision Tree** | Rule-based tree classifier |
| **Bagging Ensemble** | 10 Decision Tree estimators, 80% bootstrap sampling |

### Regression (Exact Traffic Volume)
| Model | Description |
|---|---|
| **Linear Regression** | Predicts continuous traffic volume |

### Unsupervised Learning
| Model | Description |
|---|---|
| **K-Means Clustering** | Groups data into 3 traffic clusters |

### Reinforcement Learning
| Model | Description |
|---|---|
| **Q-Learning** | Optimizes traffic signal timing based on traffic state |

---

##  Evaluation Metrics

- **Classification:** Accuracy, Precision, Recall, F1-Score
- **Regression:** MAE, RMSE, R²
- **Clustering:** Davies-Bouldin Index
- **Q-Learning:** Average cumulative reward over 1000 episodes

---

##  Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/MRafayAhmed/metro-traffic-prediction.git
cd metro-traffic-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib kagglehub
```

### 3. Download the dataset (optional — dataset already included in `data/`)
```bash
python Metrodata.py
```

### 4. Run the full ML pipeline
```bash
python model.py
```

---

##  Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `scikit-learn` | ML models and evaluation |
| `matplotlib` | Visualization and plots |
| `kagglehub` | Dataset download from Kaggle |

---

##  Visualizations Generated

Running `model.py` produces the following plots:
- Classification Accuracy Comparison (bar chart)
- F1 Score Comparison (bar chart)
- Linear Regression Performance (MAE / RMSE / R²)
- K-Means Davies-Bouldin Index
- Q-Learning Convergence Curve

  <img width="634" height="486" alt="image" src="https://github.com/user-attachments/assets/6da1b142-f88a-448c-ad04-ba2fbd2ec73f" />
<img width="623" height="466" alt="image" src="https://github.com/user-attachments/assets/befa0b14-454c-4473-a0cf-c0b42ad26b61" />
<img width="605" height="468" alt="image" src="https://github.com/user-attachments/assets/3fe821dc-6b0b-4694-a229-748a65676f28" />
<img width="615" height="467" alt="image" src="https://github.com/user-attachments/assets/b01cab20-50e3-4612-89bd-86d4428c59a3" />
<img width="609" height="446" alt="image" src="https://github.com/user-attachments/assets/4c3d56a5-4b12-4795-84b6-5ae5ea1afe06" />
<img width="616" height="457" alt="image" src="https://github.com/user-attachments/assets/14f8bbba-9a1f-4a51-b6b4-304a63bec792" />


---

## 📄 License

This project is for educational purposes. Dataset credit goes to the original authors on Kaggle.
