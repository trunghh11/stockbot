######## 3. Mô hình dự đoán đa cổ phiếu (Multi Stock Model) ##########

import sys, os
sys.path.append('../python')
from util_1 import *

'''
Other initializers of the class the can be mentioned along with the default values:
past_history = 60       # number of days in the past you want to look at
forward_look = 1        # number of days forward that you want to predict
train_test_split = 0.8
batch_size = 30
epochs = 50
steps_per_epoch = 200   #
validation_steps = 50    # Steps taken while validating over the dev set
verbose = 1             # Whether you want tensorflow to print out the training gunk
naive = False           # Should've called it a better name but it just refers to one LSTM
values = 200            # Future days that you want to plot for (computed one day at a time)
'''


ticker_dict, tickerSymbols = get_categorical_tickers()
start="2010-01-01"
end="2019-12-31"
################### Multi Stock Model với sameTickerTestTrain = FALSE ###################
tickeranalysis = ticker_dict['materials'][0] # Mã BHP
tickerList = ticker_dict['materials']
tickerList.remove(tickeranalysis) # Tách mã BHP ra khỏi ngành energy

LSTM_1 = LSTM_Model_MS(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True,
                       tickerSymbolList = tickerList, sameTickerTestTrain = False)
LSTM_1.full_workflow_and_plot()
LSTM_1.plot_bot_decision()

LSTM_2 = LSTM_Model_MS(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0,
                       tickerSymbolList = tickerList, sameTickerTestTrain = False)
LSTM_2.full_workflow_and_plot()
LSTM_2.plot_bot_decision()

LSTM_3 = LSTM_Model_MS(tickerSymbol = tickeranalysis, start = start, end = end, depth = 1,
                       tickerSymbolList = tickerList, sameTickerTestTrain = False)
LSTM_3.full_workflow_and_plot()
LSTM_3.plot_bot_decision()

plt.show()

################### Multi Stock Model với sameTickerTestTrain = TRUE ###################
tickeranalysis = ticker_dict['materials'][0]
tickerList = ticker_dict['materials']
tickerList.remove(tickeranalysis)

LSTM_4 = LSTM_Model_MS(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True,
                       tickerSymbolList = tickerList, sameTickerTestTrain = True)
LSTM_4.full_workflow_and_plot()
LSTM_4.plot_bot_decision()

LSTM_5 = LSTM_Model_MS(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0,
                       tickerSymbolList = tickerList, sameTickerTestTrain = True)
LSTM_5.full_workflow_and_plot()
LSTM_5.plot_bot_decision()

LSTM_6 = LSTM_Model_MS(tickerSymbol = tickeranalysis, start = start, end = end, depth = 1,
                       tickerSymbolList = tickerList, sameTickerTestTrain = True)
LSTM_6.full_workflow_and_plot()
LSTM_6.plot_bot_decision()

plt.show()

