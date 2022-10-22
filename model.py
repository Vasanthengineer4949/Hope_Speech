import config
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import json
import joblib



class Model():

    def __init__(self, train_df_path, test_df_path, lang):
        self.lang = lang
        self.train_df = pd.read_csv(train_df_path)
        self.test_df = pd.read_csv(test_df_path)
        try:
            self.train_df = self.train_df.drop("label", axis=1)
            self.test_df = self.test_df.drop("label", axis=1)
        except:
            pass
        print(f"{self.lang} Data Read")
        self.dependent_feature = "label_enc"
        self.X_train = self.train_df.drop(self.dependent_feature, axis=1)
        self.X_test = self.test_df.drop(self.dependent_feature, axis=1)
        self.y_train = self.train_df[self.dependent_feature]
        self.y_test = self.test_df[self.dependent_feature]
        print(f"{self.lang} Data Splitted")

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "tree": DecisionTreeClassifier(),
            "rf" : RandomForestClassifier(),
            "lgbm" : LGBMClassifier(),
            "cat" : CatBoostClassifier(),
            "xgb" : XGBClassifier(),
            "logreg" : LogisticRegression(max_iter=300),
            "svc" : SVC(gamma=0.5, C=0.1),
            "stk" : StackingClassifier(
                estimators =  [('tree', DecisionTreeClassifier()),
                ('rf', RandomForestClassifier()),
                ('lgbm', LGBMClassifier()),
                ('cat', CatBoostClassifier()),
               ],
                final_estimator=LogisticRegression(),
                stack_method='auto',
                n_jobs=-1,
                passthrough=False
            )
        }

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            print(f"{self.lang} {model} Training Completed")
            key = list(models.keys())
            print(key[i])
            model_out_path = config.MODEL_OUT_PATH+self.lang+key[i]
            y_pred = model.predict(X_test)
            y_pred = pd.DataFrame(y_pred)
            print(f"{self.lang} {model} Prediction done")
            y_pred.to_csv(model_out_path+".csv", index=False)

            metric_dict = {
                f"{self.lang} accuracy_{model}":float(accuracy_score(y_test, y_pred)),
                f"{self.lang} precision_score_{model}":float(precision_score(y_test, y_pred, average="weighted")),
                f"{self.lang} recall_score_{model}":float(recall_score(y_test, y_pred, average="weighted")),
                f"{self.lang} f1_score_{model}":float(f1_score(y_test, y_pred, average="weighted")),
                f"{self.lang} confusion_matrix_{model}":confusion_matrix(y_test, y_pred).tolist()
            }
            print(f"{self.lang} {model} Metric Calculation done")
            out_file = open(model_out_path+".json", "w")
            json.dump(metric_dict, out_file, indent = 6)
            print(f"{self.lang} {model} Metric Saving Done")
            out_file.close()

            joblib.dump(model, model_out_path+".pkl")
            print(f" {self.lang}{model} Saving Done")

    def run(self):
        self.train_model(self.X_train, self.y_train, self.X_test, self.y_test)

