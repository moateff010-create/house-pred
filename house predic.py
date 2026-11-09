import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
x_train = pd.read_csv(r"D:/datasets/house_pred/train.csv")
x_test = pd.read_csv(r"D:/datasets/house_pred/test.csv")

print(x_train.head())
print(x_train.describe())
print(x_train.info())
x_train["Alley"] = x_train["Alley"].fillna("no alley access")
x_test["Alley"] = x_test["Alley"].fillna("no alley access")
x_train["MasVnrType"] = x_train["MasVnrType"].fillna("No Mass Vnr")
x_test["MasVnrType"] = x_test["MasVnrType"].fillna("No Mass Vnr")
x_train["BsmtQual"] = x_train["BsmtQual"].fillna("No Basment")
x_test["BsmtQual"] = x_test["BsmtQual"].fillna("No Basment")
x_train["BsmtCond"] = x_train["BsmtCond"].fillna("No Basment")
x_test["BsmtCond"] = x_test["BsmtCond"].fillna("No Basment")
x_train["BsmtExposure"] = x_train["BsmtExposure"].fillna("No Basment")
x_test["BsmtExposure"] = x_test["BsmtExposure"].fillna("No Basment")
x_train["BsmtFinType1"] = x_train["BsmtFinType1"].fillna("Unknown")
x_test["BsmtFinType1"] = x_test["BsmtFinType1"].fillna("UNknown")
x_train["BsmtFinType2"] = x_train["BsmtFinType2"].fillna("No Basment")
x_test["BsmtFinType2"] = x_test["BsmtFinType2"].fillna("No Basment")
x_train["Electrical"] = x_train["Electrical"].fillna("Unknown")
x_test["Electrical"] = x_test["Electrical"].fillna("Unknown")
x_train["FireplaceQu"] = x_train["FireplaceQu"].fillna("No Fire Place")
x_test["FireplaceQu"] = x_test["FireplaceQu"].fillna("No Fire Place")
x_train["GarageType"] = x_train["GarageType"].fillna("No Garage")
x_test["GarageType"] = x_test["GarageType"].fillna("No Garage")
x_train["GarageYrBlt"] = x_train["GarageYrBlt"].fillna("No Garage")
x_test["GarageYrBlt"] = x_test["GarageYrBlt"].fillna("No Garage")
x_train["GarageFinish"] = x_train["GarageFinish"].fillna("No Garage")
x_test["GarageFinish"] = x_test["GarageFinish"].fillna("No Garage")
x_train["GarageQual"] = x_train["GarageQual"].fillna("No Garage")
x_test["GarageQual"] = x_test["GarageQual"].fillna("No Garage")
x_train["GarageCond"] = x_train["GarageCond"].fillna("No Garage")
x_test["GarageCond"] = x_test["GarageCond"].fillna("No Garage")
x_train["PoolQC"] = x_train["PoolQC"].fillna("No Pool")
x_test["PoolQC"] = x_test["PoolQC"].fillna("No Pool")
x_train["Fence"] = x_train["Fence"].fillna("No Fence")
x_test["Fence"] = x_test["Fence"].fillna("No Fence")
x_train["MiscVal"] = x_train["MiscVal"].fillna("None")
x_test["MiscVal"] = x_test["MiscVal"].fillna("None")


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

x_train = x_train[x_train["GrLivArea"] < 4000]
target = np.log1p(x_train['SalePrice'])
x_train = x_train.drop(columns=['SalePrice',"Id"], errors='ignore')
x_test = x_test.drop(columns="Id")
train_len = len(x_train)

df_full = pd.concat([x_train, x_test], axis=0)
df_full["TotalSF"] = df_full["TotalBsmtSF"] + df_full["1stFlrSF"] + df_full["2ndFlrSF"] + df_full["GrLivArea"]
df_full["HouseAge"] = df_full["YrSold"] - df_full["YearBuilt"]
df_full["TotalBath"] = df_full["FullBath"] + (0.5 * df_full["HalfBath"]) + df_full["BsmtFullBath"] +(0.5 * df_full["BsmtHalfBath"])
df_full["IsRemodeled"] = (df_full["YearBuilt"] != df_full["YearRemodAdd"]).astype(int)
num_cols = df_full.select_dtypes(include=['number']).columns
for col in num_cols:
  df_full[col] = df_full[col].fillna(df_full[col].mean())


df_full = pd.get_dummies(df_full)


X_train_final = df_full.iloc[:train_len]
X_test_final = df_full.iloc[train_len:]
y_train_final = target


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train_final, test_size=0.2, random_state=42
)

# model = RandomForestRegressor(n_estimators=100, random_state=42)

# model.fit(X_tr, y_tr)

# pred_log = model.predict(X_val)
# pred = np.expm1(pred_log)
# y_val_pred = np.expm1(y_val)
# rmse = np.sqrt(mean_squared_error(y_val_pred, pred))
# print(f'RMSE Error on Validation: {rmse}')
model = XGBRegressor(n_estimators = 300,learning_rate = 0.08,subsample = 0.8,colsample_bytree = 0.8,max_depth = 4,random_state = 42)
model.fit(X_tr,y_tr)
pred_log = model.predict(X_val)
pred = np.expm1(pred_log)
y_val_pred = np.expm1(y_val)
rmse = np.sqrt(mean_squared_error(y_val_pred,pred))
print(f'RMSE Error on Validation: {rmse}')


final_pred_log = model.predict(X_test_final)
final_pred = np.expm1(final_pred_log)
sample_sub=pd.read_csv(r'D:/datasets/house_pred/sample_submission.csv')
sample_sub['SalePrice'] = final_pred
sample_sub.to_csv(r'D:/datasets/house_pred/sample_submission.csv',index = False)
