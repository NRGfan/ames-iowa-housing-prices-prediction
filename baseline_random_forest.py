#rank 2470
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

list_of_params = ["Id", "MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "1stFlrSF", "2ndFlrSF", "YearRemodAdd", "LowQualFinSF", "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "YrSold", "MoSold"]

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test = test[list_of_params]
test = pd.get_dummies(test)

list_of_params.append("SalePrice")
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df = df[list_of_params]
df = pd.get_dummies(df)

regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
x = df.iloc [:, : -1]
y = df.iloc [:, -1 :]
y = np.ravel(y)
regressor.fit(x, y)
Y_pred = regressor.predict(test)  # test the output by changing values
# Visualising the Random Forest Regression results

ids = test.Id.values.tolist()
submission_csv = pd.DataFrame(
{"Id": ids,
 "SalePrice": Y_pred
})

submission_csv.to_csv("/kaggle/working/submission_csv.csv", index=False)
print("Exported")