# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

First we can take the dataset based on one input value and some mathematical calculus output value.Next define the neural network model in three layers.First layer has six neurons and second layer has four neurons,third layer has one neuron.The neural network model takes the input and produces the actual output using regression.

## Neural Network Model

![image](https://github.com/VishalGowthaman/basic-nn-model/assets/94165380/c47bb649-7c5e-49b9-a94e-5191c1fdb8e6)




## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```
DEVELOPED BY : VISHAL GOWTHAMAN K R
REG NO : 212221230123
``` 
```
from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('deep learning').sheet1 
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'int'})
df = df.astype({'output':'int'})

x = df[["input"]] .values
y = df[["output"]].values

scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)

ai = Seq([
    Den(8,activation = 'relu',input_shape=[1]),
    Den(15,activation = 'relu'),
    Den(1),
])

ai.compile(optimizer = 'rmsprop',loss = 'mse')

ai.fit(x_train,y_train,epochs=2000)
ai.fit(x_train,y_train,epochs=2000)

loss_plot = pd.DataFrame(ai.history.history)
loss_plot.plot()

err = rmse()
preds = ai.predict(x_test)
err(y_test,preds)
x_n1 = [[30]]
x_n_n = scaler.transform(x_n1)
ai.predict(x_n_n)
```

## Dataset Information

![image](https://github.com/VishalGowthaman/basic-nn-model/assets/94165380/14033943-d420-4b00-94dc-037cbb7baf8a)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/VishalGowthaman/basic-nn-model/assets/94165380/69222961-5956-4875-8ac3-47eeeb8017cd)


### Test Data Root Mean Squared Error

![image](https://github.com/VishalGowthaman/basic-nn-model/assets/94165380/56dce712-f3d9-414f-b90b-62e861e50486)



### New Sample Data Prediction

![image](https://github.com/VishalGowthaman/basic-nn-model/assets/94165380/f98767be-2149-42f9-bd82-8b921c48a28c)


## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
