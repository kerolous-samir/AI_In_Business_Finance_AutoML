from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from preprocess import preprocess_data

creditcard = preprocess_data("../data/1.1_UCI_Credit_Card.csv")

x_cat = creditcard[['SEX','EDUCATION','MARRIAGE']]
onehotencoder = OneHotEncoder()
x_cat = onehotencoder.fit_transform(x_cat).toarray()
x_cat = pd.DataFrame(x_cat)

x_num = creditcard[['LIMIT_BAL','AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

x_all = pd.concat([x_cat, x_num], axis=1)
minmaxscaler = MinMaxScaler()
x_all = minmaxscaler.fit_transform(x_all)

y = creditcard['default.payment.next.month']
X_train, X_test, y_train, y_test = train_test_split(x_all, y, test_size=0.2)

model = xgb.XGBClassifier(objective='reg:squarederror',learning_rate=0.1, max_depth=5, n_estimators=100)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
