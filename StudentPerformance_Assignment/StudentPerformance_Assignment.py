import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error, r2_score


data_path = r"Python/MachineLearningRepo/StudentPerformance_Assignment/student_performance.csv"
df = pd.read_csv(data_path)
df.head()