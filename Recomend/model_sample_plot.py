import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator, FixedLocator, MultipleLocator


def parse_log():
    lines = []
    pmf_results = {}
    bpmf_results = {}
    als_results = {}

    with open('log-sample.log') as file:
        lines = [line.rstrip() for line in file]
        for l in lines:
            t = re.findall(r"[-+]?\d*\.\d+|\d+", l)
            t = [int(t[0]), float(t[1]), float(t[2]), int(t[3])]
            if l[0:19] == 'INFO:recommend.pmf:':
                if t[3] in pmf_results:
                    pmf_results[t[3]].append(t)
                else:
                    pmf_results[t[3]] = []
                    pmf_results[t[3]].append(t)
            elif l[0:19] == 'INFO:recommend.bpmf':
                if t[3] in bpmf_results:
                    bpmf_results[t[3]].append(t)
                else:
                    bpmf_results[t[3]] = []
                    bpmf_results[t[3]].append(t)
            elif l[0:19] == 'INFO:recommend.als:':
                if t[3] in als_results:
                    als_results[t[3]].append(t)
                else:
                    als_results[t[3]] = []
                    als_results[t[3]].append(t)



    print(pmf_results)
    print(bpmf_results)
    print(als_results)
    return pmf_results, bpmf_results, als_results
    #return pmf_results[::1], bpmf_results[::1], als_results[::1]

pmf, bpmf, als = parse_log()

pmf_x = []
pmf_y = []
bpmf_x = []
bpmf_y = []
als_x = []
als_y = []



fig = plt.figure(figsize=(7, 4))

ax = fig.add_subplot()
plt.xlabel('Number of samples (*100000)')
plt.ylabel('RMSE')


pmf_t = []
bpmf_t = []
als_t = []

for key, val in pmf.items():
    pmf_x.append(key/100000)
    n = [n[1] for n in val]
    pmf_y.append(min(n))
    pmf_t.append(val[-1][2])


for key, val in bpmf.items():
    bpmf_x.append(key/100000)
    n = [n[1] for n in val]
    bpmf_y.append(min(n))
    bpmf_t.append(val[-1][2])


for key, val in als.items():
    als_x.append(key/100000)
    n = [n[1] for n in val]
    als_y.append(min(n))
    als_t.append(val[-1][2])






plt.title("RMSE versus the number of samples using 10D feature vectors")


for i in range(len(pmf_x)):
    if i%2==0 and i>0:
        ax.text(pmf_x[i], pmf_y[i]+0.005, str(int(pmf_t[i])) + 'sec.')
        plt.scatter(pmf_x[i], pmf_y[i], color='green', s=20, marker='o')

for i in range(len(bpmf_x)):
    if i%2==0 and i>0:
        ax.text(bpmf_x[i], bpmf_y[i]+0.006, str(int(bpmf_t[i])) + 'sec.')
        plt.scatter(bpmf_x[i], bpmf_y[i], color='red', s=20, marker='o')

for i in range(len(als_x)):
    if i%2==0 and i>0:
        ax.text(als_x[i], als_y[i]+0.008, str(int(als_t[i])) + 'sec.')
        plt.scatter(als_x[i], als_y[i], color='blue', s=20, marker='o')




ax.plot(pmf_x, pmf_y, label='PMF', color='green')


ax.plot(bpmf_x, bpmf_y, label='BPMF', color='red')

ax.plot(als_x, als_y, label='ALS', color='blue')

plt.legend()




plt.show()

