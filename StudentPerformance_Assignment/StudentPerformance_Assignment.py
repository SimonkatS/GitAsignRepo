import pandas as pd 
import numpy as np 
import seaborn as sns
import sklearn


dataset = pd.read_csv(r"Python\MachineLearningRepo\StudentPerformance_Assignment\student_performance.csv")
print(dataset.describe())
print(dataset.info())




# Clustering features to work with , dont mind it for now
X = [weekly_self_study_hours, attendance_percentage, class_participation, total_score]