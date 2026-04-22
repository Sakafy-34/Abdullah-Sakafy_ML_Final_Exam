import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None
    print("Optional dependency 'seaborn' not installed. Plots will use matplotlib. Install with: pip install seaborn")
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline




from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



from sklearn.ensemble import VotingRegressor, StackingRegressor




from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


df = pd. read_csv("insurance 2.csv")

print(df.head())

print(len(df.columns))

print(len(df))

print(df.shape)

corr_target = df.select_dtypes(include=np.number).corr()['charges'].sort_values(ascending=False)
print(corr_target)

X = df.drop('charges',axis=1)
y = df['charges']

numeric_features = X.select_dtypes(include = ['int64','float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

print(numeric_features)
print(categorical_features)



for col in numeric_features:
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3-Q1

  lower = Q1 - 1.5*IQR
  upper = Q3 + 1.5*IQR

  outliers = df[(df[col]<lower) | (df[col]>upper)]
  print(f"Numer of detected outliers in {col}: ", len(outliers))
  if(len(outliers)>0):
    df[col] = np.where(df[col]>upper,upper,df[col])
    df[col] = np.where(df[col]<lower,lower,df[col])




num_transformer = Pipeline (
    steps = [
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline( steps = [
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
] )


preprocessor = ColumnTransformer(
    transformers= [
        ('num',num_transformer,numeric_features),
        ('cat',cat_transformer,categorical_features)
    ]
)

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state=42)



reg_lr = LinearRegression()
reg_rf = RandomForestRegressor( n_estimators=100, random_state=42 )
reg_gb = GradientBoostingRegressor( n_estimators=100 , random_state=42 )



voting_reg = VotingRegressor(
    estimators= [
        ('lr', reg_lr),
        ('rf',reg_rf),
        ('gb', reg_gb)
    ]
)



stacking_reg = StackingRegressor(
    estimators= [
        ('rf',reg_rf),
        ('gb', reg_gb)
    ],
    final_estimator= Ridge() 
)




model_to_train = {
    'Linear Regression' : reg_lr,
    'Random Forest' : reg_rf,
    'Gradient Boosting': reg_gb,
    'Voting Ensemble ' : voting_reg,
    'Stacking Ensemble ' : stacking_reg

}




result = []

for name , model in model_to_train.items():
  
  pipe = Pipeline(
      [
          ('preprocessor', preprocessor),
          ('model',model)
      ]
  )

  #train

  pipe.fit(X_train,y_train)

  #predict

  y_pred = pipe.predict(X_test)

  #Evaluate

  r2 = r2_score(y_test,y_pred)
  rmse = np.sqrt(mean_squared_error(y_test,y_pred))
  mae = mean_absolute_error(y_test,y_pred)

  result.append({
      "Model": name,
      "R2 Score" :r2,
      "RMSE": rmse,
      "MAE" : mae
  })

results_df = pd.DataFrame(result).sort_values("R2 Score", ascending=False)

print(results_df)


best_model_name = results_df.iloc[0]['Model']
best_model_obj = model_to_train[best_model_name]




final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model',best_model_obj)
])

final_pipe.fit(X_train,y_train)
y_final_pred = final_pipe.predict(X_test)




plt.figure( figsize = (10,6) )

if sns is not None:
    sns.scatterplot(x=y_test, y=y_final_pred, alpha = 0.6, color='teal' )
else:
    plt.scatter(x=y_test, y=y_final_pred, alpha = 0.6, color='teal')

plt.plot( [2,5] , [2,5], color = "red", linestyle = '--'  )

plt.xlabel("Actual Charge")
plt.ylabel("Predicted Charge")

plt.grid(True)
plt.show()


from sklearn.model_selection import cross_val_score

rf_pipeline = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('model', GradientBoostingRegressor(n_estimators=100 , random_state=42))

     ]

)



cv_scores = cross_val_score( rf_pipeline,X_train,y_train,cv=10, scoring='neg_mean_squared_error' )
cv_rmse = np.sqrt(-cv_scores)

print(cv_rmse)

print(cv_rmse.mean())

print(cv_rmse.std())

rf_pipeline = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('model', GradientBoostingRegressor(n_estimators=100 , random_state=42))

     ]

)


param_grid = {
    'model__n_estimators' : [100,200] ,
    'model__max_depth': [None,10,20],
    'model__min_samples_split' : [2,5]
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator = rf_pipeline,
    param_grid = param_grid,
    cv = 10 ,
    scoring = 'neg_root_mean_squared_error',
    n_jobs =-1,
    verbose = 2

)

grid_search.fit(X_train,y_train)

print(-grid_search.best_score_)

print(grid_search.best_params_)


filename = "gradient_boosting_model.pkl"

with open( filename, "wb" ) as file:
    pickle.dump( grid_search, file )


with open( filename, "rb" ) as file:
    gb_loaded_model = pickle.load(file)

y_pred_final = gb_loaded_model.predict(X_test)
print(y_pred_final)

r2 = r2_score(y_test,y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test,y_pred_final))
mae = mean_absolute_error(y_test,y_pred_final)
final_result = []
final_result.append({
      "R2 Score" :r2,
      "RMSE": rmse,
      "MAE" : mae
})
final_result_df = pd.DataFrame(final_result)
print(final_result_df)