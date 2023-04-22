import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

#See the yahoo finance ticker for your stock symbol
stock_symbol = 'mrf.ns'
user_input = st.text_input("Enter the stock ticker symbol", 'GOOG')

# data = yf.download(tickers=stock_symbol,period='5y',interval='1d')
startDate = st.date_input(
    "Start Date For Forcasting",
    datetime.date(2019, 7, 6))
st.write('Start Date For Forcasting', startDate)
st.write('End Date For Forcasting', date.today() )
start = startDate
end =date.today() 
df = yf.download(
    user_input,
    start ,
    end ,
    progress = False
    )

if st.button('Show Data'):
    # st.write(df)
    close = df[['Close']]

    plt.figure(figsize=(12,6))
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.legend()


    ds = close.values


    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    #Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test and train data sizes
    train_size = int(len(ds_scaled)*0.70)
    test_size = len(ds_scaled) - train_size

    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]


    #creating dataset in time series for LSTM model 
    #X[100,120,140,160,180] : Y[200]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)


    #Taking 100 days price as one record for training
    time_stamp = 100
    X_train, y_train = create_ds(ds_train,time_stamp)
    X_test, y_test = create_ds(ds_test,time_stamp)

    #Reshaping data to fit into LSTM model
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)





    #Creating LSTM model using keras
    # model = Sequential()
    # model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    # model.add(LSTM(units=50,return_sequences=True))
    # model.add(LSTM(units=50))
    # model.add(Dense(units=1,activation='linear'))
    # model.summary()


    #Training model with adam optimizer and mean squared error loss function
    # model.compile(loss='mean_squared_error',optimizer='adam')
    # model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)

    #PLotting loss, it shows that loss has decreased significantly and model trained well
    # loss = model.history['loss']
    # plt.plot(loss)

    #Saving model
    # model.save('stock_model.h5')

    #Loading model
    model = load_model('stock_model.h5')

    #Predicitng on train and test data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    #Inverse transform to get actual value
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)

    #Comparing using visuals
    plt.plot(normalizer.inverse_transform(ds_scaled))
    plt.plot(train_predict)
    plt.plot(test_predict)


    test = np.vstack((train_predict,test_predict))

    #Combining the predited data to create uniform data visualization
    plt.plot(normalizer.inverse_transform(ds_scaled))
    plt.plot(test)


    #Getting the last 100 days records
    ans3= len(ds_test)-100
    fut_inp = ds_test[ans3:]

    fut_inp = fut_inp.reshape(1,-1)

    tmp_inp = list(fut_inp)

    #Creating list of the last 100 data
    tmp_inp = tmp_inp[0].tolist()


    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
        

    # print(lst_output)


    #Creating a dummy plane to plot graph one after another
    plot_new=np.arange(1,101)
    plot_pred=np.arange(101,131)

    st.header('Prediction Chart')
    fig2 = plt.figure(figsize=(12,6))
    ans = len(ds_scaled) - 100
    graph1 = plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[ans:]))
    graph2 =plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
    st.pyplot(fig2)



    ds_new = ds_scaled.tolist()
    #Entends helps us to fill the missing value with approx value
    ds_new.extend(lst_output)
    ans2 =len(ds_new)-31
    plt.plot(ds_new[ans2:])


    #Creating final data for plotting
    final_graph = normalizer.inverse_transform(ds_new).tolist()

    #Plotting final results with predicted value after 30 Days
    st.subheader('Next 30 Days Prediction Chart')
    fig3 = plt.figure(figsize=(12,6))
    plt.plot(final_graph , label = 'Closing Price')
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.title("{0} prediction of next month close".format(user_input))
    plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
    plt.legend()

    st.pyplot(fig3)





    st.subheader('Closing Price vs Time Chart with 50MA & 200MA')

    st.markdown("When the :red[50-day moving average] is above the **:blue[200-day moving average]** , it is considered a bullish signal, indicating that the stock price may continue to rise. ")
    st.markdown("When the :red[50-day moving average] is below the **:blue[200-day moving average]** , it is considered a bearish signal, indicating that the stock price may continue to fall.")

    ma200 = df.Close.rolling(200).mean()
    ma50 = df.Close.rolling(50).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma50 ,'r', label = '50-day moving average')
    plt.plot(ma200 ,'b' ,label = '200-day moving average')
    plt.plot(df.Close ,'g') 

    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

else:
    st.write("Please enter a valid stock ticker")

