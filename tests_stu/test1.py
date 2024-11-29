######### 1. So sánh mô hình LSTM đơn lớp (single layer LSTM) và LSTM 2 lớp (double layer LSTM) ########
import sys, os
sys.path.append('../python')
from util import *   # Các giá trị mặc định, hàm train model, test model,...

'''
Các giá trị mặc định sẽ sử dụng (có thể thay đổi trong quá trình viết mã):
past_history = 60       # Số ngày trong quá khứ muốn xem xét (sử dụng 60 ngày gần nhất để dự đoán các ngày tiếp theo)
forward_look = 1        # Mô hình sẽ dự đoán giá của từng ngày một
train_test_split = 0.8  # Tỉ lệ train / test: 80 / 20
batch_size = 30         # Batch_size = 30
epochs = 50             # Số epochs = 50
steps_per_epoch = 200   # Số bước trên mỗi epoch = 200
validation_steps = 50    # Số bước được thực hiện khi xác nhận (validation) trên tập phát triển (dev set)
verbose = 1             # In ra chi tiết từng epoch khi huấn luyện
naive = False           # Chỉ định xem có sử dụng một mô hình LSTM đơn giản (naive LSTM) hay không
values = 200            # Đây là số ngày tương lai mà mô hình sẽ dự đoán và vẽ biểu đồ. (200 ngày)
'''


ticker_dict, tickerSymbols = get_categorical_tickers()
start="2010-01-01"
end="2019-12-31"
##############################################
tickeranalysis = tickerSymbols[1] # Mã GOOG

# LSTM_1 = LSTM_Model_with_Indicators(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True)
# LSTM_1.full_workflow_and_plot()
# LSTM_1.plot_bot_decision()

LSTM_1 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True)
LSTM_1.full_workflow_and_plot()
LSTM_1.plot_bot_decision()
plt.clf()

# LSTM_2 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0)
# LSTM_2.full_workflow_and_plot()
# LSTM_2.plot_bot_decision()
# plt.clf()

# LSTM_3 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 1)
# LSTM_3.full_workflow_and_plot()
# LSTM_3.plot_bot_decision()

# plt.show()
##############################################
# tickeranalysis = tickerSymbols[1] # Mã GOOG
# LSTM_1 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True)
# LSTM_1.full_workflow_and_plot()
# plt.clf()
# LSTM_1.plot_bot_decision()
# plt.clf()
# LSTM_2 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0)
# LSTM_2.full_workflow_and_plot()
# plt.clf()
# LSTM_2.plot_bot_decision()
# plt.clf()
# plt.show()
##############################################
# tickeranalysis = tickerSymbols[2] # Mã MSFT
# LSTM_1 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True)
# LSTM_1.full_workflow_and_plot()
# plt.clf()
# LSTM_1.plot_bot_decision()
# plt.clf()
# LSTM_2 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0)
# LSTM_2.full_workflow_and_plot()
# plt.clf()
# LSTM_2.plot_bot_decision()
# plt.clf()