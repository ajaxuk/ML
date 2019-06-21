import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

# Linear
#y = 2*(x) + 3
#y_noise = 2 * np.random.normal(size=x.size)

# Cubic
y = 1*(x**3) + 1*(x**2) + 1*x  +3
y_noise = 20 * np.random.normal(size=x.size)

# Quadratic
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)


# Exponential
y = np.exp(x)
y_noise = 0

# Logarithmic
y = np.log(x)
y_noise = 0

# Signoidal / Logistic
y = 1-4/(1+np.power(3,x-2))

ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

df = pd.read_csv("china_gdp.csv")
df.head(10)


# Visualise the data to decide on likely best model to fit
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Build model using Sigmoid / Logistic
def sigmoid (x,Beta_1,Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y

# Sample Sigma that might fit
beta_1 = 0.10
beta_2 = 1990.0

Y_pred = sigmoid(x_data,beta_1,beta_2)

plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.show()

# Normalise the x and y data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)


# Use curve-fit to find parameters
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Evaluate model using Train / Test

# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )


