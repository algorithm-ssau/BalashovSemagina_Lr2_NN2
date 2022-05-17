
import pandas as pd

import matplotlib as mpl

import IPython.display

import tensorflow as tf

from WindowGenerator import WindowGenerator, Baseline

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

PATH_DS = 'NFLX.csv'
df = pd.read_csv(PATH_DS)
date_time = pd.to_datetime(df.pop('Date'), format='%Y-%m-%d')
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]
num_features = df.shape[1]

# Нормализация датасета
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')


label_columns = 'Volume'

OUT_STEPS = 30
multi_window = WindowGenerator(input_width=OUT_STEPS, label_width=OUT_STEPS, shift=OUT_STEPS,
                               label_columns=[label_columns], train_df=train_df, val_df=val_df, test_df=test_df)

print(multi_window)

multi_lstm_model = tf.keras.Sequential([

    tf.keras.layers.LSTM(units=50, return_sequences=False),
    
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])

])

MAX_EPOCHS = 100


def compile_and_fit(model, window, patience=MAX_EPOCHS):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history


history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()
multi_val_performance = {}
multi_performance = {}
multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(model=multi_lstm_model, plot_col=label_columns)
