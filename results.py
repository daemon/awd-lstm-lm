from matplotlib import pyplot as plt
import numpy as np

results = """10,251.15
20,92.43
30,74.51
40,67.85
50,64.64
60,62.64
70,61.60
80,60.94
90,60.40
100,60.11"""

results2 = """10,346.74
20,124.98
30,80.37
40,69.07
50,65.03
60,62.95
70,61.78
80,60.96
90,60.50
100,60.11"""

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

plt.title("Markov Dropout Pruning on QRNN for PTB")
plt.ylabel("Word-level Perplexity")
plt.xlabel("% FLOPs in QRNN layers")
plt.xlim(0)
y_range = (59, 94)
plt.ylim(*y_range)
# plt.yticks(np.arange(54, 75, 1))
plt.yticks(np.arange(*y_range, 1))
plt.xticks(np.arange(0, 110, 10))
plt.hlines(60.6, 0, 110, color="red", linewidth="1")
plt.text(93, 61, "Official GitHub result")
plt.show()