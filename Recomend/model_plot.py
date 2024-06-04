import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator, FixedLocator, MultipleLocator


def parse_log():
    lines = []
    pmf_results = []
    bpmf_results = []
    als_results = []

    with open('log10-30.log') as file:
        lines = [line.rstrip() for line in file]
        for l in lines:
            #print(l[0:19])
            if l[0:19] == 'INFO:recommend.pmf:':
                t = re.findall(r"[-+]?\d*\.\d+|\d+", l)
                t = [int(t[0]), float(t[1]), float(t[2])]
                pmf_results.append(t)
            elif l[0:19] == 'INFO:recommend.bpmf':
                t = re.findall(r"[-+]?\d*\.\d+|\d+", l)
                t = [int(t[0]), float(t[1]), float(t[2])]
                bpmf_results.append(t)
            elif l[0:19] == 'INFO:recommend.als:':
                t = re.findall(r"[-+]?\d*\.\d+|\d+", l)
                t = [int(t[0]), float(t[1]), float(t[2])]
                als_results.append(t)




    print(pmf_results)
    print(bpmf_results)
    print(als_results)
    return pmf_results, bpmf_results, als_results
    #return pmf_results[::1], bpmf_results[::1], als_results[::1]

pmf, bpmf, als = parse_log()



pmf_x = [n[0] for n in pmf]
pmf_y = [n[1] for n in pmf]


bpmf_x = [n[0] for n in bpmf]
bpmf_y = [n[1] for n in bpmf]


als_x = [n[0] for n in als]
als_y = [n[1] for n in als]



fig = plt.figure(figsize=(7, 4))


ax = fig.add_subplot()

plt.title("Performance of PMF, BPMF, ALS using 10D feature vectors")


plt.xlabel('Epochs')
plt.ylabel('RMSE')

for i in range(len(pmf_x)):
    if i%5==0 and i>4:
        ax.text(pmf_x[i], pmf_y[i]+0.005, str(int(pmf[i][2])) + 'sec.')
        plt.scatter(pmf_x[i], pmf_y[i], color='green', s=20, marker='o')

for i in range(len(bpmf_x)):
    if i%5==0 and i>4:
        ax.text(bpmf_x[i], bpmf_y[i]+0.005, str(int(bpmf[i][2])) + 'sec.')
        plt.scatter(bpmf_x[i], bpmf_y[i], color='red', s=20, marker='o')

for i in range(len(als_x)):
    if i%5==0 and i>4:
        ax.text(als_x[i], als_y[i]+0.005, str(int(als[i][2])) + 'sec.')
        plt.scatter(als_x[i], als_y[i], color='blue', s=20, marker='o')



ax.plot(pmf_x, pmf_y, label='PMF', color='green')

ax.plot(bpmf_x, bpmf_y, label='BPMF', color='red')

ax.plot(als_x, als_y, label='ALS', color='blue')



ax.yaxis.set_major_locator(FixedLocator([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]))

ax.xaxis.set_major_locator(MultipleLocator(base=2))

plt.legend()

plt.show()

