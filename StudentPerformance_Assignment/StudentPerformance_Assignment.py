import pandas as pd 
import numpy as np 
import seaborn as sns


dataset = pd.read_csv(r"Python\MachineLearningRepo\StudentPerformance_Assignment\student_performance.csv")
print(dataset.describe())
print(dataset.info())