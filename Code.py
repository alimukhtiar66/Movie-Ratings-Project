import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

Dataset = pd.read_csv('dataset.csv')
Dataset.dropna(inplace=True)

features = ['% Action', '% Thriller', '% Drama', '% Acting Quality',
            '% Comedy', '% Romance', '% Sci-Fi', 'Average Review Score']
target = 'Predicted Rating'

X = Dataset[features]
y = Dataset[target]
#X = Dataset.iloc[:,[0,1,2,3,4,5,7]]
#y = Dataset.iloc[:,8]

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)

model = LinearRegression()

model.fit(X_train, y_train)

#y_pred = model.predict(X_test)
h_theta = model.predict(X_test)

j_theta = mean_squared_error(y_test, h_theta)
r2_s = r2_score(y_test, h_theta)

print("The Mean Squared Error is:", j_theta)
print("The R square Score is:", r2_s)

