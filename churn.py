import numpy as np
import pandas as pd

from dataprep.eda import create_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #LabelEncoder

from sklearn.tree import DecisionTreeClassifier #머신러닝 시작
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost

from churnDBMD import *
from churnMLMD import *

import warnings; warnings.filterwarnings("ignore")


dfpath = ""
user = ""
pw = ""
host = ""
dbName = ""
tbName = ""

df = readTb(host, user, pw, dbName, tbName) #각 인자 별도 지정하여 전달

report = create_report(df)
report.save("telecom.html")

print(df.info()) #데이터확인: 타입 이상함

objDf = df.select_dtypes("object") #범주형 분리
numDf = df.select_dtypes("number") #수치형 분리
objcols = list(objDf)
numcols = list(numDf)
print("object columns count", len(objcols))
print("numerical columns count", len(numcols))

plt.figure(figsize=(10, 20)) #범주형 분포확인
plotCols(df, 6, 3, False, objcols) #TotalCharges: 연속형이 맞다고 판단
plotCols(df, 1, 3, False, numcols) #seniorcitizen: 범주형이 맞다고 판단

df.columns = df.columns.str.lower()
df["seniorcitizen"] = df["seniorcitizen"].astype("object") #타입변환
df.loc[df[df["totalcharges"]==" "].index, "totalcharges"] = np.NaN
df["totalcharges"] = df["totalcharges"].astype("float")
df = df.dropna() #로우 삭제
print("preprocessed df shape:", df.shape)
print("changed seniorcitizen dtype", df["seniorcitizen"].dtype)
print("changed totalcharges dtype", df["totalcharges"].dtype)


objcols2 = list(df.select_dtypes("object")) #다시그려보자
numcols2 = list(df.select_dtypes("number"))

plt.figure(figsize=(10, 20))
plotCols(df, 6, 3, False, objcols2)
plotCols(df, 1, 3, False, numcols2)


plt.figure(figsize=(10, 20)) #연속형 변수와 타겟값을 살펴보자
plotCols(df, 6, 3, True, objcols2)

plt.figure(figsize=(20, 7))
plotCols(df, 1, 3, True, numcols2)


df = df.iloc[:, 1:] #필요없는 행 제거
objcols = list(df.select_dtypes("object"))
df[objcols] = df[objcols].apply(LabelEncoder().fit_transform) #레이블링
df[objcols] = df[objcols].astype("int")
df.head()


corr = df.corr() #상관관계
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask, 1)] = True
plt.figure(figsize=(10, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm", linewidths=.3, cbar_kws={"shrink": .5})


X = df.drop(["churn"], axis=1) #머신러닝 시작: 데이터분리
y = df["churn"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

dtclf = DecisionTreeClassifier(random_state=0)
lrclf = LogisticRegression(random_state=0)
vtclf = VotingClassifier([("decision", dtclf), ("logistic", lrclf)], voting="soft")
rfclf = RandomForestClassifier(random_state=0)
xgbclf = xgboost.XGBClassifier()

models = [dtclf, lrclf, vtclf, rfclf, xgbclf]

trainsize = np.linspace(.1, 1.0, 5)
cv=3

plotRocCurve(models[0], x_train, x_test, y_train, y_test, trainsize, cv)
plotRocCurve(models[1], x_train, x_test, y_train, y_test, trainsize, cv)
plotRocCurve(models[2], x_train, x_test, y_train, y_test, trainsize, cv)
plotRocCurve(models[3], x_train, x_test, y_train, y_test, trainsize, cv)
plotRocCurve(models[4], x_train, x_test, y_train, y_test, trainsize, cv)