import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from collections import Counter
import re

# Text styles
RESET = "\033[0m"
BOLD = "\033[1m"

# Text colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"

# Example usage
print(f"\t\t\t\t\t{BOLD}{RED}PREDICTIONS{RESET}",end=' ')
print(f"{GREEN}HUB{RESET}")
print("Press 1 for predicting house price")
print("Press 2 for predicting customer churn")
print("Press 3 for predicting rainfall")
print("Press 4 for predicting emails' authentication")
print("Press 5 for exit the program")

choice = input("::::?")

if choice == '1':
    # implementing linear regression
    dataset = pd.read_csv("J:\\4-Fourth Smester\\coding_semester4\\AI Lab\\ML\\1-2-3-4-5-Project\\Housing.csv")
    print(dataset)
    X = dataset[['area','bedrooms','bathrooms','stories','mainroad','guestroom',
                'basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']]
    # X['furnishingstatus'] = X['furnishingstatus'].values.str.replace({"furnished":2,"semi-furnished":1,"unfurnished":0})
    X.iloc[X['furnishingstatus'] == 'furnished'] = 2
    X.iloc[X['furnishingstatus'] == 'unfurnished'] = 0
    X.iloc[X['furnishingstatus'] == 'semi-unfurnished'] = 1

    X.iloc[X['mainroad'] == 'yes'] = 1
    X.iloc[X['mainroad'] == 'no'] = 0

    X.iloc[X['guestroom'] == 'yes'] = 1
    X.iloc[X['guestroom'] == 'no'] = 0

    X.iloc[X['basement'] == 'yes'] = 1
    X.iloc[X['basement'] == 'no'] = 0

    X.iloc[X['hotwaterheating'] == 'yes'] = 1
    X.iloc[X['hotwaterheating'] == 'no'] = 0

    X.iloc[X['airconditioning'] == 'yes'] = 1
    X.iloc[X['airconditioning'] == 'no'] = 0

    X.iloc[X['prefarea'] == 'yes'] = 1
    X.iloc[X['prefarea'] == 'no'] = 0




    Y = dataset[['price']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)


    model = LinearRegression()
    model.fit(X_train, Y_train)

    print("\t\t\tEnter the house details please which you want to buy")
    area = int(input("Enter area:"))
    bedrooms = int(input("Enter number of bedrooms:"))
    bathrooms = int(input("Enter number of bathrooms:"))
    stories = int(input("Enter number of stories:"))
    mainroad = int(input("Do you want to buy house on main road(0 for no 1 for yes):"))
    guestroom = int(input("Do you want to have guestroom:"))
    stories = int(input("Do you want to have basement:"))
    hotwaterheating = int(input("Do you want to have hotwaterheating:"))
    airconditioning = int(input("Do you want to have airconditioning:"))
    parking = int(input("Enter number of parkings:"))
    prefarea = int(input("Do you want to have prefarea:"))
    furnishingstatus = int(input("Enter furnishing level furnished: 2, semi-furnished:1,unfurnished:0 :"))

    input_feature = [[area,bedrooms,bathrooms,stories,mainroad,
                     guestroom,stories,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus]]
    input_pred = pd.DataFrame(input_feature)
    
    price = model.predict(input_pred)
    price = round(price[0][0],2)
    print("The house price is:","PKR")
    accuracy = model.score(X_test, Y_test)
    accuracy = round(accuracy,2)
    print("Our Model Accuracy is:", accuracy * 100,"%")


elif choice == '2':
    dataset = pd.read_csv("J:\\4-Fourth Smester\\coding_semester4\\AI Lab\\ML\\1-2-3-4-5-Project\\customer_churn.csv")
    # pd.set_option('display.max_columns',None)
    print(dataset)
    X = dataset[['Age','Total_Purchase','Account_Manager','Years','Num_Sites']]
    Y = dataset[['Churn']]


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000) # if number of iterations of model are too less to predict data, we can 
    # manually add

    model.fit(X_train, Y_train.values.ravel()) # we use ravel() to flatten the array



    Age = float(input("Enter the age of the customer:"))
    Total_Purchase = float(input("Enter the total purchase of the customer:"))
    Account_Manager = int(input("Is he Account Manager(1 for yes and 0 for no):"))
    Years = float(input("Enter the years spent by customer:"))
    Num_Sites = float(input("Enter the number of sites that customer visited:"))

    input_feature = [[Age, Total_Purchase, Account_Manager, Years, Num_Sites]]
    input_pred = pd.DataFrame(input_feature, columns=['Age', 'Total_Purchase', 'Account_Manager','Years', 'Num_Sites'])


    prediction = model.predict(input_pred)
    print("Prediction (0 = No Churn, 1 = Churn):", prediction[0])

    accuracy = model.score(X_test, Y_test)
    accuracy = round(accuracy,2)
    print("Our Model Accuracy is:",accuracy * 100,"%")


elif choice == '3':
    # dataset = pd.read_csv("J:\\4-Fourth Smester\\coding_semester4\\AI Lab\\ML\\1-2-3-4-5-Project\\weather.csv")
    # pd.set_option('display.max_columns',None)
    # print(dataset)
    # X = dataset[['p (mbar)', 'T (degC)', 'rh (%)']]
    # Y = dataset[['rain']]
    # p = float(input("Enter pressure in milibars:"))
    # T = float(input("Enter the temperature in celsius:"))
    # rh = float(input("Enter rh percentage:"))
    pass


elif choice == '4':
    # Load the dataset
    data = pd.read_csv("J:\\4-Fourth Smester\\coding_semester4\\AI Lab\\ML\\1-2-3-4-5-Project\\emails.csv")

    # Prepare features and labels
    X = data.drop(columns=['Email No.', 'Prediction']).values  # All word count columns
    y = data['Prediction']  # The labels (0 for not spam, 1 for spam)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Function to count words in the email text
    def count_words(email_text, word_list):
        # Clean the text: remove punctuation and convert to lower case
        email_text = re.sub(r'[^\w\s]', '', email_text.lower())
        words = email_text.split()
        
        # Count occurrences of each word in the word_list
        word_counts = Counter(words)
        
        # Create a list of counts for the specified word_list
        counts = [word_counts[word] for word in word_list]
        return counts

    # Function to predict if an input email is spam or not
    def predict_spam(email_text):
        # Define the words you want to track (same as your dataset columns)
        word_list = data.columns[1:-1].tolist()  # Adjust based on your dataset
        
        # Count words in the input email text
        word_counts = count_words(email_text, word_list)
        
        # Make prediction
        prediction = model.predict([word_counts])
        return 'Spam' if prediction[0] == 1 else 'Not Spam'

    # Example usage for user input
    user_input_email = input("Enter the email text: ")
    result = predict_spam(user_input_email)
    print(f'The email is: {result}')
elif choice == '5':
    exit