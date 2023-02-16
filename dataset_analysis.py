import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import optuna
from sklearn.preprocessing import LabelEncoder


df_train = pd.read_csv(f"dataset/train.csv", index_col= "PassengerId")
df_test = pd.read_csv(f"dataset/test.csv", index_col= "PassengerId")
gender = pd.read_csv(f"dataset/gender_submission.csv", index_col= "PassengerId")

#df_train = pd.get_dummies(df_train, columns = ['Sex', 'Embarked'])
#df_test = pd.get_dummies(df_test, columns = ['Sex', 'Embarked'])

# Label encode Embarked and Sex feature
label_encoder = LabelEncoder()
df_train['Sex'] = label_encoder.fit_transform(df_train['Sex'])
df_train['Embarked'] = label_encoder.fit_transform(df_train['Embarked'])
df_test['Sex'] = label_encoder.fit_transform(df_test['Sex'])
df_test['Embarked'] = label_encoder.fit_transform(df_test['Embarked'])

df_train = df_train.reset_index()
df_test = df_test.reset_index()
handle_train = df_train.drop(['Survived', 'Cabin','Name','PassengerId','Ticket','Sex','Embarked'], axis = 1)
handle_test = df_test.drop(['Cabin', 'Name','PassengerId','Ticket','Sex','Embarked'], axis = 1)
imputer = KNNImputer(n_neighbors=10)
imputer.fit(handle_train)

df_train[handle_train.columns] = imputer.transform(handle_train)
df_test[handle_test.columns] = imputer.transform(handle_test)

df_train["Cabin"] = df_train.Cabin.map(lambda x: 1 if isinstance(x, str) else 0)
df_test["Cabin"] = df_test.Cabin.map(lambda x: 1 if isinstance(x, str) else 0)

feature = df_train.drop(['Survived','PassengerId','Name','Ticket'], axis = 1)
label = df_train['Survived']
test = df_test.drop([ 'Name','PassengerId','Ticket'], axis = 1)

scaler = StandardScaler()
scaler.fit(feature)
feature = scaler.transform(feature)
test = scaler.transform(test)

X_train, X_test, y_train, y_test = train_test_split(
      feature,label, test_size=0.10, random_state=42)
'''
rfc = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=7, max_features='auto', n_jobs= -1 )

param_grid = { 
    'n_estimators': [200, 500, 1000],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8, 10, 15, 20, 25, 30],
    'criterion' :['gini', 'entropy', 'log_loss']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, verbose = 3)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)

rfc_opt = RandomForestClassifier(n_estimators=CV_rfc.best_params_['n_estimators'], criterion=CV_rfc.best_params_['criterion'], max_depth=CV_rfc.best_params_['max_depth'], max_features=CV_rfc.best_params_['max_features'], n_jobs= -1 )
rfc_opt.fit(feature, y_train)

pred=rfc_opt.predict(X_test)
print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


def objective(trial):
    logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e1, log=True)
    max_iter = trial.suggest_int("max_iter", 50, 3000)
    solver = trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"])
    penalty = trial.suggest_categorical("penalty", ["l2"])

    LRC = LogisticRegression(C=logreg_c,max_iter=max_iter,solver=solver,penalty=penalty)
    LRC.fit(X_train, y_train)
    return 1.0 - accuracy_score(y_test, LRC.predict(X_test))
study = optuna.create_study()
study.optimize(objective, n_trials = 500)
print(study.best_params)
print(1.0 - study.best_value)

hyperparameters = {
'C':np.logspace(0, 10, 50), 
'penalty':['l2'],
'random_state':[42],
'max_iter':[200,500,1000],
'solver':['newton-cg', 'lbfgs', 'liblinear']
}
modellrOpt = LogisticRegression(C=study.best_params.get('logreg_c'),max_iter=study.best_params.get('max_iter'),solver=study.best_params.get('solver'),penalty=study.best_params.get('penalty'))

#rfOpt.fit(X_train, y_train)

#pred=rfOpt.predict(X_test)
#print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))
modellrOpt.fit(feature, label)
predictions = modellrOpt.predict(test)

output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
'''
def objective(trial):
    max_depth = trial.suggest_int("max_depth", 2, 300)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 300)
    max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 300))
    splitter = trial.suggest_categorical("splitter",["best","random"])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 300)
    DTC = DecisionTreeClassifier(min_samples_split = min_samples_split, 
                            max_leaf_nodes = max_leaf_nodes, max_depth=max_depth, random_state=42, splitter=splitter,min_samples_leaf=min_samples_leaf,
                            criterion = criterion)
    DTC.fit(X_train, y_train)
    return 1.0 - accuracy_score(y_test, DTC.predict(X_test))
study = optuna.create_study()
study.optimize(objective, n_trials = 200)
print(study.best_params)
print(1.0 - study.best_value)

dtOpt = DecisionTreeClassifier(min_samples_split = study.best_params.get('min_samples_split'), 
                                max_leaf_nodes = study.best_params.get('max_leaf_nodes'), max_depth=study.best_params.get('max_depth'), random_state=42, splitter=study.best_params.get('splitter'),min_samples_leaf=study.best_params.get('min_samples_leaf'),
                                criterion = study.best_params.get('criterion'))

dtOpt.fit(feature, label)
predictions = dtOpt.predict(test)

output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")