import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report

#API used :   https://www.kaggle.com/datasets/goyalshalini93/car-data

data_path = r'Ypoligistikh_Noim\CarDataset_Assignment.csv'#document paths xriazodai r gia na min einai invalid syntax 
#copy to relative path tou katevasmenou arxeiou (wtf)

try: 
    car_data = pd.read_csv(data_path)
    feature = input("Give me now :: ")
    
except FileNotFoundError:
    print('Shit')
    
    
def compute_statistics(car_data, feature):
    stats = {
        "mean": car_data[feature].mean(),
        "std": car_data[feature].std(),
        "median": car_data[feature].median(),
    }
    
    return stats    

def print_stats(car_data , feature):
    
    stats = compute_statistics(car_data, feature)
    print(f" {feature}: {stats}")
    
print_stats(car_data, feature)



