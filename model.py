import pandas as pd
import numpy as np

# Load dataset (update path when you add file to VS Code)
df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")

# View data
print(df.head())
print(df.info())

# Check missing values
print(df.isnull().sum())

# Fill missing values (if any)
df = df.ffill()

# Remove duplicates
df = df.drop_duplicates()

# Convert date_time column
df['date_time'] = pd.to_datetime(df['date_time'])

# Extract useful time features
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour
df['dayofweek'] = df['date_time'].dt.dayofweek

# Create rush hour feature
df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 19) else 0)

# Weekend feature
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# Weather severity mapping (simple encoding)
weather_map = {
    'Clear': 0,
    'Clouds': 1,
    'Rain': 2,
    'Snow': 3,
    'Mist': 2,
    'Haze': 2,
    'Drizzle': 2,
    'Thunderstorm': 3
}

df['weather_severity'] = df['weather_main'].map(weather_map)
df['weather_severity'] = df['weather_severity'].fillna(1)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['weather_description'] = le.fit_transform(df['weather_description'])
df['holiday'] = le.fit_transform(df['holiday'])
df['weather_main'] = le.fit_transform(df['weather_main'])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

num_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']

df[num_cols] = scaler.fit_transform(df[num_cols])

# Target variable
target = 'traffic_volume'

# Features
X = df.drop(columns=[target, 'date_time'])
y = df[target]

print(X.shape, y.shape)

#Models IMplementation

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create classification target
traffic_bins = pd.qcut(y, q=3, labels=[0, 1, 2])  # 0=Low, 1=Med, 2=High
y_class = traffic_bins
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Initialize model
knn = KNeighborsClassifier(n_neighbors=5)
# Train model
knn.fit(X_train_c, y_train_c)
# Predict
y_pred_knn = knn.predict(X_test_c)
# Evaluate
print("KNN Accuracy:", accuracy_score(y_test_c, y_pred_knn))

#SVM
from sklearn.svm import SVC
# Initialize SVM model
svm = SVC(kernel='rbf')
# Train model
svm.fit(X_train_c, y_train_c)
# Predict
y_pred_svm = svm.predict(X_test_c)
# Evaluate
print("SVM Accuracy:", accuracy_score(y_test_c, y_pred_svm))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Initialize model
dt = DecisionTreeClassifier(random_state=42)
# Train model
dt.fit(X_train_c, y_train_c)
# Predict
y_pred_dt = dt.predict(X_test_c)
# Evaluate
print("Decision Tree Accuracy:", accuracy_score(y_test_c, y_pred_dt))


#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Initialize model
lr = LinearRegression()
# Train model
lr.fit(X_train, y_train)
# Predict
y_pred_lr = lr.predict(X_test)
# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))


#K-Mean Clustering
from sklearn.cluster import KMeans
# Initialize KMeans (choose 3 clusters for traffic levels)
kmeans = KMeans(n_clusters=3, random_state=42)
# Fit model (unsupervised)
kmeans.fit(X)
# Assign cluster labels
clusters = kmeans.labels_
# Add clusters to dataframe
df['cluster'] = clusters
print(df[['traffic_volume', 'cluster']].head())


#Q-learning
import numpy as np
# States: Low, Medium, High traffic
states = [0, 1, 2]
# Actions: 0 = short green, 1 = medium green, 2 = long green
actions = [0, 1, 2]
# Q-table initialization
Q = np.zeros((len(states), len(actions)))

# Learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000
for episode in range(episodes):
    state = np.random.choice(states)
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(actions)  # explore
    else:
        action = np.argmax(Q[state])  # exploit
    # Reward system (simulated logic)
    if state == 2 and action == 2:
        reward = 10
    elif state == 1 and action == 1:
        reward = 5
    else:
        reward = -1
    next_state = np.random.choice(states)
    # Q-learning update
    Q[state, action] = Q[state, action] + alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state, action]
    )
print("Trained Q-Table:\n", Q)



#Ensemble Learning(Bagging)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Base model (weak learner)
base_model = DecisionTreeClassifier(random_state=42)
# Bagging model
bagging_model = BaggingClassifier(
    estimator=base_model,   # model to repeat
    n_estimators=10,        # number of trees/models
    max_samples=0.8,        # 80% data per model
    bootstrap=True,         # sampling with replacement
    random_state=42
)
bagging_model.fit(X_train_c, y_train_c)
y_pred_bag = bagging_model.predict(X_test_c)
print("Bagging Accuracy:", accuracy_score(y_test_c, y_pred_bag))


#Classification Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def evaluate_classification(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall   :", recall_score(y_true, y_pred, average='weighted'))
    print("F1 Score :", f1_score(y_true, y_pred, average='weighted'))
# Evaluate all models
evaluate_classification("KNN", y_test_c, y_pred_knn)
evaluate_classification("SVM", y_test_c, y_pred_svm)
evaluate_classification("Decision Tree", y_test_c, y_pred_dt)
evaluate_classification("Bagging", y_test_c, y_pred_bag)


#Regression Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
print("\nLinear Regression Performance:")
print("MAE :", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2  :", r2_score(y_test, y_pred_lr))


#Clustering Evaluation
from sklearn.metrics import davies_bouldin_score
db_index = davies_bouldin_score(X, clusters)
print("\nK-Means Clustering Performance:")
print("Davies-Bouldin Index:", db_index)


#Q-Learning Evaluation
# Track rewards
rewards = []
for episode in range(episodes):
    state = np.random.choice(states)
    total_reward = 0
    for step in range(10):
        action = np.argmax(Q[state])        
        if state == 2 and action == 2:
            reward = 10
        elif state == 1 and action == 1:
            reward = 5
        else:
            reward = -1       
        total_reward += reward
        state = np.random.choice(states)
    rewards.append(total_reward)
print("\nAverage Reward:", np.mean(rewards))

#Convergence Check
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.title("Q-Learning Convergence")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 📌 CLASSIFICATION PLOTS
# =========================

models = ["KNN", "SVM", "Decision Tree", "Bagging"]

accuracy = [
    accuracy_score(y_test_c, y_pred_knn),
    accuracy_score(y_test_c, y_pred_svm),
    accuracy_score(y_test_c, y_pred_dt),
    accuracy_score(y_test_c, y_pred_bag)
]

f1_scores = [
    f1_score(y_test_c, y_pred_knn, average='weighted'),
    f1_score(y_test_c, y_pred_svm, average='weighted'),
    f1_score(y_test_c, y_pred_dt, average='weighted'),
    f1_score(y_test_c, y_pred_bag, average='weighted')
]

# Accuracy Comparison
plt.figure()
plt.bar(models, accuracy)
plt.title("Classification Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

# F1 Score Comparison
plt.figure()
plt.bar(models, f1_scores)
plt.title("F1 Score Comparison")
plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.show()


# =========================
# 📌 REGRESSION PLOT
# =========================

mae = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2 = r2_score(y_test, y_pred_lr)

metrics = ["MAE", "RMSE", "R2"]
values = [mae, rmse, r2]

plt.figure()
plt.bar(metrics, values)
plt.title("Linear Regression Performance")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.show()


# =========================
# 📌 CLUSTERING PLOT
# =========================

plt.figure()
plt.bar(["K-Means"], [db_index])
plt.title("Davies-Bouldin Index (Lower is Better)")
plt.ylabel("Score")
plt.show()


# =========================
# 📌 Q-LEARNING PLOT
# =========================

plt.figure()
plt.plot(rewards)
plt.title("Q-Learning Convergence")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.show()