import pandas as pd 
import numpy as np 
import seaborn as sns


df = pd.read_csv(r"Python\MachineLearningRepo\StudentPerformance_Assignment\student_performance.csv")
print(df.describe())