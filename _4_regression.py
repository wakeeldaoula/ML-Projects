    # Manual Method
# import numpy as np
# import pandas as pd

#     # Load the dataset
# dataset = pd.read_csv("https://bit.ly/drinksbycountry")

#     # Display the dataset
# print(dataset)

#     # Prepare data
# x = dataset['wine_servings']
# y = dataset['spirit_servings']

#     # Calculate averages
# x_avg = np.average(x)
# y_avg = np.average(y)

#     # Calculate deviations
# x_dev = [i - x_avg for i in x]
# y_dev = [j - y_avg for j in y]

#     # Calculate products of deviations
# prod_dev = [x_dev[i] * y_dev[i] for i in range(len(x_dev))]

#     # Calculate sum of products and sum of squared deviations
# sum_prod_dev = sum(prod_dev)
# sum_sqr_x = sum(i ** 2 for i in x_dev)

#     # Calculate slope (m) and intercept (b)
# m = sum_prod_dev / sum_sqr_x
# b = y_avg - (m * x_avg)

#     # Print the regression equation
# print(f"The regression equation is: y = {m}x + ({b})")

#     # User input for predictions
# while True:
#     choice = int(input("Enter 1 for finding y, 2 for finding x, 0 for exiting the program: "))
    
#     if choice == 1:
#         x_value = int(input("Enter value of x (wine servings): "))
#         y_value = m * x_value + b
#         print("Predicted value of spirit servings:", y_value)
        
#     elif choice == 2:
#         y_value = int(input("Enter value of y (spirit servings): "))
#         if m == 0:
#             print("Cannot calculate x, as slope m is zero.")
#         else:
#             x_value = (y_value - b) / m
#             print("Predicted value of wine servings:", x_value)
            
#     elif choice == 0:
#         print("Exiting the program.")
#         break
        
#     else:
#         print("Invalid choice. Please try again.")



import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 random points in the range [0, 2]
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with some noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model parameters
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()