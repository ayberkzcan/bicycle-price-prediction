import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

dataframe = pd.read_excel("bisiklet_fiyatlari.xlsx")
print(dataframe.head())

sbn.pairplot(dataframe)
plt.show()

y = dataframe["Fiyat"].values
x = dataframe[["BisikletOzellik1","BisikletOzellik2"]].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)
y_train.shape
y_test.shape

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")
model.summary()
model.fit(x_train, y_train, epochs=500, verbose=0)


loss = model.history.history["loss"]

sbn.lineplot(x=range(len(loss)),y=loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Eğitim Kaybı Grafiği")
plt.show()

trainloss = model.evaluate(x_train,y_train,verbose=0)
testloss = model.evaluate(x_test,y_test,verbose=0)
testtahminleri = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, testtahminleri))
print("RMSE:", rmse)

print("Train Loss:", trainloss)
print("Test Loss:", testloss)

tahmindf = pd.DataFrame(y_test,columns=["Gerçek Y"])

testtahminleri = pd.Series(testtahminleri.reshape(-1,))

tahmindf = pd.concat([tahmindf,testtahminleri],axis=1)
tahmindf.columns = ["Gerçek Y","Tahmin Y"]
print(tahmindf.head())

sbn.scatterplot(x="Gerçek Y", y="Tahmin Y",data=tahmindf)
plt.title("Gerçek Fiyat vs Tahmin Edilen Fiyat")
plt.show()

mae = mean_absolute_error(tahmindf["Gerçek Y"],tahmindf["Tahmin Y"])
mse = mean_squared_error(tahmindf["Gerçek Y"],tahmindf["Tahmin Y"])
print("MAE:", mae)
print("MSE:", mse)
print(dataframe.describe())

print("Please enter values between 1745 and 1754 for reliable predictions.")
oz1 = float(input("özellik1 = "))
oz2 = float(input("özellik2 = "))
yenibisikletözellikleri = [[oz1, oz2]]
yenibisikletözellikleri = scaler.transform(yenibisikletözellikleri)
tahmin = model.predict(yenibisikletözellikleri)
print("Tahmini fiyat =", round(tahmin[0][0],2))
