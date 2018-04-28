from matplotlib import pyplot as plt
import numpy as np

results = """10,246.02
20,94.92
30,74.58
40,67.89
50,64.48
60,62.34
70,61.11
80,60.46
90,60.08
100,59.69"""

results2 = """10,349.14
20,131.37
30,82.86
40,69.59
50,64.82
60,62.54
70,61.36
80,60.47
90,60.02
100,59.69"""

# channel pruning on full QRNN
results3 = """40,254.30
50,170.77
60,115.21
70,87.67
80,70.23
90,61.48
100,58.62"""

# MD on LSTM, no rescale
results_lstm = """60,210.48
70,95.03
80,69.75
90,61.17
100,58.05"""

# CP on LSTM, no rescale
results_lstm_cp = """50,122.59
60,101.96
70,80.45
80,65.89
90,60.51
100,58.05"""

# 40% FLOPs train from scratch: 62.67 test PPL
# 40% FLOPs train from 40% linear dropout: 61.85 test PPL
# 60% FLOPs train from scratch: 60.39 test PPL
# 60% FLOPs train from 60% linear dropout: 59.70 test PPL

def plot(data):
    x_values = []
    y_values = []
    for line in data.split("\n"):
        point = [float(v) for v in line.split(",")]
        x_values.append(point[0])
        y_values.append(point[1])
        # plt.hlines(y_values, 0, x_values, color="black", linewidth="1", linestyle="dashed")
    plt.plot(x_values, y_values, "-*")

plot(results)
plot(results2)
plot(results3)
plot(results_lstm)
plot(results_lstm_cp)

plt.title("Markov Dropout Pruning on QRNN for PTB")
plt.ylabel("Word-level Perplexity")
plt.xlabel("% FLOPs in RNN layers")
plt.xlim(0)
y_range = (58, 94)
plt.ylim(*y_range)
# plt.yticks(np.arange(54, 75, 1))
plt.yticks(np.arange(*y_range, 1))
plt.xticks(np.arange(0, 110, 10))
plt.hlines(58.62, 0, 110, color="red", linewidth="1")
plt.text(93, 58.82, "Original model result")
plt.show()
