import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.stats.proportion import proportion_confint


f = open("n17res", "r")


# returns P coverageProb rmse rbeta
def parseLine(s):
    return np.array([float(pair.split('=')[1]) for pair in s.split(' ')])[[1, 2, 3, 4]]


contents = f.read().splitlines()

data = np.array(list(map(parseLine, contents)))
Ps = data[:, 0]
prob = data[:, 1]
rmse = data[:, 2]
rbeta = data[:, 3]

confint = [proportion_confint(int(p*1280), 1280, method='wilson') for p in prob]
lower, upper = ((zip(*confint)))
lower = np.array(lower)
upper = np.array(upper)

matplotlib.pyplot.show()
plt.show(block=True)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('log2(P)')
ax1.set_ylabel('Coverage probability', color=color)
ax1.errorbar(np.log2(Ps), prob, yerr = np.vstack([prob - lower, upper - prob]), color=color, elinewidth=0.5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.hlines(0.95, np.min(np.log2(Ps)), np.max(np.log2(Ps)), linestyles='dashed', colors=(1,0.7,0.7))

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average RMSE and r_beta')
ax2.plot(np.log2(Ps), rmse, color=color)
ax2.plot(np.log2(Ps), rbeta, color="tab:green")





plt.show()