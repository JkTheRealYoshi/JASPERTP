import pandas as pd
import base64
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from binance.client import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from flask import Flask, request, jsonify, render_template
from keras.models import Sequential
from numpy import ndarray
from pyalgotrade.technical import bollinger
from pyalgotrade.technical import macd
from pyalgotrade.technical import rsi
from pyalgotrade.technical import stoch
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor , RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct , WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D , MaxPooling1D
from tensorflow.keras.layers import Input , Dense , Flatten
from tensorflow.keras import Sequential

api_key = 'yI7lZJ9VTIQmMf8gSPUzp2HA8I2BgtyBIEZvc7ZPaoYKblqzTlLbGflGFEPZf9xA'
api_secret = 'PEQqk26zJR6cmCAz6NyjIlnJO0xX45yUYl1QbPy28WM0oeguOP5TPsHMlzeVriEq'

app = Flask(__name__)

def set_weight(user_input):
    if user_input == 'risk management':
        weight_estimation = 0.65
        weight_prediction = 0.35
    else:
        weight_estimation = 0.35
        weight_prediction = 0.65

        return weight_estimation , weight_prediction

    binance_instance = Client(api_key , api_secret)

    default_symbols = "BTC/USDT,ETH/USDT,BNB/USDT,XRP/USDT,DOGE/USDT,TRON/USDT"
    timeframe_options = ["15m" , "1h" , "4h" , "1d"]

def fetch_crypto_data(symbol , timeframe , limit):
    ohlcv = binance_instance.get_Klines(symbol , timeframe , limit=limit)
    df = pd.DataFrame(ohlcv , columns=['timestamp' , 'open' , 'high' , 'low' , 'close' , 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'] , unit='ms')
    df.set_index('timestamp' , inplace=True)

    return df

def apply_technical_indicators(df , timeframe):
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values

    if timeframe == '1d':
        rsi_period = 14
        stoch_period = (5 , 3 , 3)
        bbands_period = 20
        macd_periods = (12 , 26 , 9)

    rsi_indicator = rsi.RSI(close_prices , rsi_period)
    df['RSI'] = pd.Series(rsi_indicator , index=df.index)

    stoch_indicator = stoch.StochasticOscillator(high_prices , low_prices , close_prices , stoch_period[0] ,
                                                 stoch_period[1] , stoch_period[2])
    df['slowk'] , df['slowd'] = pd.Series(stoch_indicator.getSlowK() , index=df.index) , pd.Series(
        stoch_indicator.getSlowD() , index=df.index)

    bbands_indicator = bollinger.BollingerBands(close_prices , bbands_period)
    df['upper'] , df['middle'] , df['lower'] = pd.Series(bbands_indicator.getUpperBand() , index=df.index) , pd.Series(
        bbands_indicator.getMiddleBand() , index=df.index) , pd.Series(bbands_indicator.getLowerBand() , index=df.index)

    macd_indicator = macd.MACD(close_prices , macd_periods[0] , macd_periods[1] , macd_periods[2])
    df['macd'] , df['macdsignal'] , df['macdhist'] = pd.Series(macd_indicator.getMACD() , index=df.index) , pd.Series(
        macd_indicator.getSignal() , index=df.index) , pd.Series(macd_indicator.getHistogram() , index=df.index)

    return df

def apply_fibonacci_levels(df):
    high = df['high'].max()
    low: object = df['low'].min()
    assert isinstance(low ,object )
    diff = high - low

    df['FIB_RETRACEMENT'] = df['close'].apply(lambda x: ((high - x) / diff))
    df['FIB_EXTENSION'] = df['close'].apply(lambda x: ((x - low) / diff))

    return df

def prepare_data(df , window_size):
    X , y = [] , []
    for i in range(window_size , len(df)):
        X.append(df.iloc[i - window_size:i][
                     ['RSI' , 'slowk' , 'slowd' , 'upper' , 'middle' , 'lower' , 'macd' , 'macdsignal' , 'macdhist' ,
                      'FIB_RETRACEMENT' , 'FIB_EXTENSION']].values)
        y.append(df.iloc[i]['close'])

        X , y = np.array(X) , np.array(y)
    return X , y

class XGBRegressor:
    pass

def create_xception_model(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='mse')

    return model

def train_models(X_train: object , y_train: object) -> object:
    input_shape = (X_train.shape[1] , X_train.shape[2])

    svm_model = svm.SVR(kernel='linear' , C=1 , gamma=0.1)
    svm_model.fit(X_train , y_train)

    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=32 , kernel_size=3 , activation='relu' , input_shape=input_shape))
    cnn_model.add(Conv1D(filters=32 , kernel_size=3 , activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1))
    cnn_model.compile(loss='mse' , optimizer='adam')
    cnn_model.fit(X_train , y_train , epochs=10 , verbose=0)

    gbm_model = GradientBoostingRegressor(random_state=0)
    gbm_model.fit(X_train.reshape(X_train.shape[0] , -1) , y_train)

    rf_model = RandomForestRegressor(random_state=0)
    rf_model.fit(X_train.reshape(X_train.shape[0] , -1) , y_train)

    ae_model = Sequential()
    ae_model.add(Dense(50 , activation='relu' , input_dim=X_train.shape[1]))
    ae_model.add(Dense(25 , activation='relu'))
    ae_model.add(Dense(50 , activation='sigmoid'))
    ae_model.compile(optimizer='adam' , loss='mse')
    ae_model.fit(X_train , X_train , epochs=10 , verbose=0)

    lstm_model = Sequential()
    lstm_model.add(Sequential(50 , input_shape=input_shape))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mse' , optimizer='adam')
    lstm_model.fit(X_train , y_train , epochs=10 , verbose=0)

    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel , random_state=0)
    gpr.fit(X_train.reshape(X_train.shape[0] , -1) , y_train)

    xgb_model = XGBRegressor()
    xgb_model.fit(X_train.reshape(X_train.shape[0] , -1) , y_train)

    def create_drl(input_shape):
        model = Sequential()
        model.add(Dense(64 , input_shape=input_shape , activation='relu'))
        model.add(Dense(32 , activation='relu'))
        model.add(Dense(16 , activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse' , optimizer='adam')
        return model

    rnn_model = Sequential()
    rnn_model.add(Sequential(50 , return_sequences=True , input_shape=input_shape))
    rnn_model.add(Sequential(50 , return_sequences=True))
    rnn_model.add(Sequential(50))
    rnn_model.add(Dense(1))
    rnn_model.compile(loss='mse' , optimizer='adam')
    rnn_model.fit(X_train , y_train , epochs=10 , verbose=0)

    def create_gan(input_shape):
        generator = Sequential()
        generator.add(Dense(64 , input_shape=(input_shape[0] ,)))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Dense(128))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Dense(input_shape[1] , activation='tanh'))

        discriminator = Sequential()
        discriminator.add(Dense(128 , input_shape=(input_shape[1] ,)))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(64))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(1 , activation='sigmoid'))

        gan_input = Input(shape=(input_shape[0] ,))
        gan_output = discriminator(generator(gan_input))
        gan = Model(gan_input , gan_output)
        discriminator.compile(loss='binary_crossentropy' , optimizer='adam')
        gan.compile(loss='binary_crossentropy' , optimizer='adam')
        return gan , generator , discriminator

    def create_gbt():
        model: GradientBoostingRegressor = GradientBoostingRegressor(random_state=0)

        return model

        xception_input_shape = (X_train.shape[1] , X_train.shape[2] , 1)
        xception_model = create_xception_model(xception_input_shape)
        xception_model.fit(X_train , y_train , epochs=10 , verbose=0)

        return svm_model , cnn_model , gbm_model , rf_model , ae_model , lstm_model , gpr , xgb_model , create_drl , rnn_model , create_gan , create_gbt , xception_model

def combine_predictions(X , models , weights):
    svm_model , cnn_model , gbm_model , rf_model , ae_model , lstm_model , gpr , xgb_model , drl_model , rnn_model , gan_model , gbt_model , xception_model = models

    svm_y_pred = svm_model.predict(X)
    cnn_y_pred = cnn_model.predict(X.reshape(X.shape[0] , X.shape[1] , 1))
    gbm_y_pred = gbm_model.predict(X.reshape(X.shape[0] , -1))
    rf_y_pred = rf_model.predict(X.reshape(X.shape[0] , -1))
    ae_y_pred = ae_model.predict(X.reshape(X.shape[0] , -1))
    lstm_y_pred = lstm_model.predict(X.reshape(X.shape[0] , X.shape[1] , 1))
    gpr_y_pred = gpr.predict(X.reshape(X.shape[0] , -1))
    xgb_y_pred = xgb_model.predict(X.reshape(X.shape[0] , -1))
    drl_y_pred = drl_model.predict(X)
    rnn_y_pred = rnn_model.predict(X.reshape(X.shape[0] , X.shape[1] , 1))
    gan_y_pred = gan_model[1].predict(gan_model[0].predict(X))
    gbt_y_pred = gbt_model.predict(X.reshape(X.shape[0] , -1))

    y_pred: object = svm_y_pred * weights[0] + cnn_y_pred * weights[1] + gbm_y_pred * weights[2] + \
                     rf_y_pred * weights[3] + ae_y_pred * weights[4] + lstm_y_pred * weights[5] + \
                     gpr_y_pred * weights[6] + xgb_y_pred * weights[7] + drl_y_pred * weights[8] + \
                     rnn_y_pred * weights[9] + gan_y_pred * weights[10] + gbt_y_pred * weights[11]

    return y_pred

asset = input("Enter the asset (e.g., BTC/USDT): ")
timeframe = input("Enter the timeframe (e.g., 1h, 4h, 1d): ")
limit = int(input("Enter the number of data points (limit): "))

df = fetch_crypto_data(asset , timeframe , limit)
df = apply_technical_indicators(df , timeframe)
df = apply_fibonacci_levels(df)

X , y = prepare_data(df , 50)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

svm_model , cnn_model , gbm_model , rf_model , ae_model , lstm_model , gpr , xgb_model , create_drl , rnn_model , create_gan , create_gbt, xception_model = train_models(
    X_train , y_train)

weights = [0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.05 , 0.05 , 0.05 , 0.05]
svm_y_pred = svm_model.predict(X_test)
cnn_y_pred = cnn_model.predict(X_test.reshape(X_test.shape[0] , X_test.shape[1] , 1))
gbm_y_pred = gbm_model.predict(X_test.reshape(X_test.shape[0] , -1))
rf_y_pred = rf_model.predict(X_test.reshape(X_test.shape[0] , -1))
ae_y_pred = ae_model.predict(X_test.reshape(X_test.shape[0] , -1))
lstm_y_pred = lstm_model.predict(X_test.reshape(X_test.shape[0] , X_test.shape[1] , 1))
gpr_y_pred = gpr.predict(X_test.reshape(X_test.shape[0] , -1))
xgb_y_pred = xgb_model.predict(X_test.reshape(X_test.shape[0] , -1))
drl_y_pred = create_drl(X_test.shape[1:]).predict(X_test)
rnn_y_pred = rnn_model.predict(X_test.reshape(X_test.shape[0] , X_test.shape[1] , 1))
gan_y_pred = create_gan(X_test.shape[1:])[0].predict(X_test)
gbt_y_pred = create_gbt().predict(X_test.reshape(X_test.shape[0] , -1))
xception_y_pred = xception_model.predict(X_test.reshape(X_test.shape[0] , X_test.shape[1], X_test.shape[2], 1))

y_pred = svm_y_pred * weights[0] + cnn_y_pred * weights[1] + gbm_y_pred * weights[2] + \
         rf_y_pred * weights[3] + ae_y_pred * weights[4] + lstm_y_pred * weights[5] + \
         gpr_y_pred * weights[6] + xgb_y_pred * weights[7] + drl_y_pred * weights[8] + \
         rnn_y_pred * weights[9] + gan_y_pred * weights[10] + gbt_y_pred * weights[11] + \
         xception_y_pred * weights[12]

mse = mean_squared_error(y_test , y_pred)
print("MSE:" , mse)

def calculate_trade_probability_helper(y_pred , last_close , entry , tp , sl , option):
    if option == 'long':
        target_profit = entry + tp
        stop_loss = entry - sl
    elif option == 'short':
        target_profit = entry - tp
        stop_loss = entry + sl
    else:
        raise ValueError("Invalid option. Choose either 'long' or 'short'.")

    y_pred_diff = np.diff(y_pred)

    if option == 'long':
        favorable_moves = (y_pred_diff > 0) & (y_pred[1:] >= entry * 0.99)
    elif option == 'short':
        favorable_moves = (y_pred_diff < 0) & (y_pred[1:] <= entry * 1.01)

    prob = favorable_moves.sum() / len(y_pred_diff)
    tp_suggest: Union[int, float, complex, ndarray] = np.percentile(y_pred_diff , 50)

    return prob , tp_suggest

import io

def generate_chart_image(df , y_pred , y_true , asset , timeframe , option):
    fig: object
    fig , ax = plt.subplots(figsize=(8 , 4))

    ax.plot(df.index , y_pred , label='Predicted')
    ax.plot(df.index , y , label='True')
    ax.set_title(f'{asset} - {timeframe} - {option.capitalize()} Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf , format='png')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')

    plt.close(fig)
    return img


from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()

    asset = req_data['asset']
    entry = float(req_data['entry'])
    tp = float(req_data['tp'])
    sl = float(req_data['sl'])
    option = req_data['option']

    timeframes = ['1d', '4h', '1h', '15m']
    output = []
    for timeframe in timeframes:
        df = fetch_crypto_data(asset, timeframe, 200)
        df = apply_technical_indicators(df, timeframe)
        X, _ = prepare_data(df, 50)

        cnn_y_pred = cnn_model.predict(X)
        svm_y_pred = svm_model.predict(X.reshape(X.shape[0], -1))
        gbm_y_pred = gbm_model.predict(X.reshape(X.shape[0], -1))
        rf_y_pred = rf_model.predict(X.reshape(X.shape[0], -1))
        ae_y_pred = ae_model.predict(X)
        lstm_y_pred = lstm_model.predict(X)
        gpr_y_pred = gpr.predict(X.reshape(X.shape[0], -1))
        xgb_y_pred = xgb_model.predict(X.reshape(X.shape[0], -1))

        y_pred = combine_predictions(X, (svm_model, cnn_model, gbm_model, rf_model, ae_model, lstm_model, gpr, xgb_model), weights)

        last_close = df['close'].iloc[-1]
        prob, tp_suggest = calculate_trade_probability_helper(y_pred, last_close, entry, tp, sl, option)

        img = generate_chart_image(df, y_pred, df['close'], asset, timeframe, option)

        output.append({
            'timeframe': timeframe,
            'prediction': {
                'probability': prob,
                'chart': img,
            },
            'probability': prob,
            'tp_suggest': tp_suggest
        })

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True, port=7200)
