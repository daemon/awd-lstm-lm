from matplotlib import pyplot as plt
import numpy as np

results = """20,107.05
30,83.83
40,76.15
50,72.07
60,69.69
70,68.20
80,67.42
90,66.95
100,66.47"""

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

def plot(data):
    x_values = []
    y_values = []
    for line in data.split("\n"):
        point = [float(v) for v in line.split(",")]
        x_values.append(point[0])
        y_values.append(point[1])
    plt.hlines(y_values, 0, x_values, color="black", linewidth="1", linestyle="dashed")
    plt.plot(x_values, y_values, "-*")

plot(results)
# plot(results2)
# plot(results3)
# plot(results_lstm)
# plot(results_lstm_cp)

plt.title("Linear Dropout Pruning on QRNN for WT-2")
plt.ylabel("Word-level Perplexity")
plt.xlabel("% FLOPs in RNN layers")
plt.xlim(0)
y_range = (65, 94)
plt.ylim(*y_range)
# plt.yticks(np.arange(54, 75, 1))
plt.yticks(np.arange(*y_range, 1))
plt.xticks(np.arange(0, 110, 10))
plt.hlines(66.76, 0, 110, color="red", linewidth="1")
plt.text(93, 66.96, "Original model result")
plt.show()
