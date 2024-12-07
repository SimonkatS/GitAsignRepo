import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error, r2_score

#Simon Katsaliakis 2122258
# original from kaggle before changes https://www.kaggle.com/datasets/goyalshalini93/car-data
#csv used :   https://github.com/SimonkatS/GitAsignRepo changed the CarName column to CarBrand and CarName
# Would have done GUI but had personal deadline for Friday so went for the easier option CLI 

data_path = r'Ypoligistikh_Noim\CarDataset_Assignment.csv'#document paths xriazodai r gia na min einai invalid syntax 
#copy to relative path tou katevasmenou arxeiou (wtf)

try: 
    car_data = pd.read_csv(data_path)
    
    # print(car_data['fueltype'].unique())
except FileNotFoundError:
    print('Dataset Error')
    exit()
    
price_categ = {'Low Price': 0, 'High Price': 0}# Initialization 

feat_for_cat = ['CarBrand','CarName','fueltype','carbody','cylindernumber','aspiration','doornumber','drivewheel','enginelocation','enginetype','fuelsystem']
label_encoders = {}        
for col in feat_for_cat:      # Label Encoding non numerical categories in the csv to be used in model training, predictions and some other minor things 
    le = LabelEncoder()
    car_data[f'{col}_encoded'] = le.fit_transform(car_data[col])
    label_encoders[col] = le     ################       <<<<----------------important for encoding the user inputs
label_mappings = {col: dict(zip(le.classes_, range(len(le.classes_)))) for col, le in label_encoders.items()} 


#all five do the same thing for a different input... anything else failed to go through the def . Assigning the inputs to a function(features) and making one def for all with a "for col in features" didnt work  
def brand_encoded(user_input):
    try:
        # Find the encoded value
        encoded_value = label_encoders['CarBrand'].transform([user_input])[0]
        print(f"Encoded Value for '{user_input}': {encoded_value}")
        return encoded_value
    except ValueError:
        print(f"'{user_input}' not found in CarBrand. Available options: {list(le.classes_)}")
        return None
def name_encoded(user_input):
    try:
        # Find the encoded value
        encoded_value = label_encoders['CarName'].transform([user_input])[0]
        print(f"Encoded Value for '{user_input}': {encoded_value}")
        return encoded_value
    except ValueError:
        print(f"'{user_input}' not found in CarName. Available options: {list(le.classes_)}")
        return None
def fuel_encoded(user_input):
    try:
        # Find the encoded value
        encoded_value = label_encoders['fueltype'].transform([user_input])[0]
        print(f"Encoded Value for '{user_input}': {encoded_value}")
        return encoded_value
    except ValueError:
        print(f"'{user_input}' not found in fueltype. Available options: {list(le.classes_)}")
        return None
def body_encoded(user_input):
    try:
        # Find the encoded value
        encoded_value = label_encoders['carbody'].transform([user_input])[0]
        print(f"Encoded Value for '{user_input}': {encoded_value}")
        return encoded_value
    except ValueError:
        print(f"'{user_input}' not found in carbody. Available options: {list(le.classes_)}")
        return None
def cylinder_encoded(user_input):
    
    try:
        # Find the encoded value
        encoded_value = label_encoders['cylindernumber'].transform([user_input])[0]
        print(f"Encoded Value for '{user_input}': {encoded_value}")
        return encoded_value
    except ValueError:
        print(f"'{user_input}' not found in cylindernumber. Available options: {list(le.classes_)}")
        return None

# Low and High thresholds for categorizing the prices after price prediction
low_threshold = car_data['price'].quantile(0.20) 
high_threshold = car_data['price'].quantile(0.75)
price_categ = {'Low Price': low_threshold, 'High Price': high_threshold}
print(f'Price Thresholds : Low={low_threshold:.2f}, High={high_threshold:.2f}')

def model_train(car_data): # Training the model . Used RandomForest because it was the most precise out of randomforest ,linear and dicisiontree
    
    categor_features = ['CarBrand','CarName','fueltype','carbody','cylindernumber','aspiration','doornumber','drivewheel','enginelocation','enginetype','fuelsystem']
    car_data_encoded = car_data.copy()
    for col in categor_features:                                           #might be redundant but when i take it out,  and use the one from the start of the code , it breaks 
        car_data_encoded[col] = le.fit_transform(car_data_encoded[col])
    
    X = car_data_encoded[['CarBrand','CarName','fueltype','carbody','cylindernumber']]    #setting the input
    y = car_data['price']                                                                 #setting the target
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42).fit(X_train,y_train)
    
    
    prediction = model.predict(X_test)                                    # test prediction (pray to god everytime that it works and that its a reasonable number)
    rmse = root_mean_squared_error(y_test, prediction)                    #   and give some info about the prediction
    r2 = r2_score(y_test, prediction)
    print(f'\nRoot Mean Squared Error : {rmse}')     
    print(f'R-squared : {r2}')                         #aiming for 0.7 - 0.9
    print(f"Predicted Price from Tests with default inputs: ${prediction[5]:.2f}")
    
    return model,car_data


def model_predict(car_feat,model):       # function for the prediction choise using the trained model with the user input and giving its placement in the price categories
    
    predicted_price = model.predict(car_feat)
    print(f"Predicted Price: ${predicted_price[0]:.2f}")     
    
    if predicted_price[0] < low_threshold:      #calculating price category and printing 
        category = "Good price/bargain"
    elif predicted_price[0] < high_threshold:
        category = "Mediocre price"
    else:
        category = "Expensive Car"
    print(f"Car Price Category: {category}")
    

def compute_statistics(car_data, feature):    #function for statistics about characteristics, if for numerical or nominal 
    if feature in ['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']:
        stats = {
            "~mean": round(car_data[feature].mean(), 3),
            "~std": round(car_data[feature].std(), 3),
            "~median": round(car_data[feature].median(), 3),
        }
        print(f"\nStats for given characteristic( {feature} ) :: {stats} \n")
    else:
        values = car_data[feature].value_counts()
        print(f'{values}')
        
               
def user_interface(car_data):   #  The User Interface :) CLI
    
    model,car_data = model_train(car_data)
    
    while True:  # for continual menu
        
            print("\nView Statistics for a category [ 1 ]")
            print("Price prediction by giving car features [ 2 ]")
            print("4. Exit from program [ 3 ]")
            choice = input("\n Choose a option :: ")
            
            if choice == '1':  # Statistics about Characteristics (1st part of assignment)
                
                categories = ['CarBrand','CarName','fueltype','carbody','cylindernumber','aspiration','doornumber','drivewheel','enginelocation','enginetype','fuelsystem','wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']
                print("characteristics to choose from :: ", categories)
                feature = input("\nType characteristic here :: ")
                compute_statistics(car_data, feature)
                
                
            elif choice == '2': #getting input for the prediction. Went for the simplest to understand characteristics 
                
                print("Answer the following questions with lowercase letters only (would recomment to have te csv open to match them to existing ones[breaks if not in]):")
                brand = input("give a car brand : ").lower()
                carname = input("give a car model : ").lower()         
                fuel = input("give a fuel-type : ").lower()
                body = input("give a body type : ").lower()
                cylinders = input("give a number of cylinders : ").lower()
                #         Encoding by matching the existing encoding of the csv ,     The most complicated thing in the assignment , debugging and finding fixes was very anoying 
                brand_enc = brand_encoded(brand)
                carname_enc = name_encoded(carname)
                fuel_enc = fuel_encoded(fuel)
                body_enc = body_encoded(body)
                cylinders_enc = cylinder_encoded(cylinders)
                car_feat = [[brand_enc,carname_enc,fuel_enc,body_enc,cylinders_enc]]
                model_predict(car_feat,model)         #calling the prediction function
                      
            elif choice == '3':
                
                print("exited... thanks for the patience  :')  ")
                break
            else: 
                print("Error>> not a choice try again")
                
                
user_interface(car_data) # calling the user interface to start the Code





