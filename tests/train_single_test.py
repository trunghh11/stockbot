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
verbose = 0             # Whether you want tensorflow to print out the training gunk
depth = 1               # Depth of the stacked LSTM (I could get rid of naive in the future)
naive = False           # Should've called it a better name but it just refers to one LSTM
values = 200            # Future days that you want to plot for (computed one day at a time)
'''


ticker_dict, tickerSymbols = get_categorical_tickers()
start="2010-01-01"
end="2019-12-31"
##############################################
tickeranalysis = tickerSymbols[0]
LSTM_1 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True)
LSTM_1.full_workflow_and_plot()
plt.clf()
LSTM_1.plot_bot_decision()
plt.clf()
LSTM_2 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0)
LSTM_2.full_workflow_and_plot()
plt.clf()
LSTM_2.plot_bot_decision()
plt.clf()
plt.show()
# LSTM_3 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 1)
# LSTM_3.full_workflow_and_plot()
# plt.clf()
# LSTM_3.plot_bot_decision()
# plt.clf()
# ############################################
# tickeranalysis = tickerSymbols[1]
# LSTM_4 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True)
# LSTM_4.full_workflow_and_plot()
# plt.clf()
# LSTM_4.plot_bot_decision()
# plt.clf()
# LSTM_5 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0)
# LSTM_5.full_workflow_and_plot()
# plt.clf()
# LSTM_5.plot_bot_decision()
# plt.clf()
# LSTM_6 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 1)
# LSTM_6.full_workflow_and_plot()
# plt.clf()
# LSTM_6.plot_bot_decision()
# plt.clf()
# #####################################
# tickeranalysis = tickerSymbols[2]
# LSTM_7 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True)
# LSTM_7.full_workflow_and_plot()
# plt.clf()
# LSTM_7.plot_bot_decision()
# plt.clf()
# LSTM_8 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0)
# LSTM_8.full_workflow_and_plot()
# plt.clf()
# LSTM_8.plot_bot_decision()
# plt.clf()
# LSTM_9 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 1)
# LSTM_9.full_workflow_and_plot()
# plt.clf()
# LSTM_9.plot_bot_decision()
# plt.clf()
# plt.show()
