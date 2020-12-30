import pandas as pd
import numpy as np
import math

store = dict({})
dataset = pd.read_csv("dataset.csv")
for i in dataset.columns:
    store.update({i: np.array(dataset[i])})
rows = len(store['RID'])
count = 0
for i in store['Class: buys computer']:
    if (i == 'yes'):
        count += 1
no = rows - count
p_yes = count / rows
p_no = no / rows


def log(num, base):
    if (num == 0):
        num = 1
    return math.log(num, base)


info_D = -1 * p_yes * log(p_yes, 2) - 1 * p_no * log(p_no, 2)
info_A = ({})
gain = ({})
del store['RID']
l = store['Class: buys computer']
del store['Class: buys computer']
for attr in store:
    info_A[attr] = 0
    gain[attr] = 0
    for val in set(store[attr]):
        count1, count2 = 0, 0
        for ind, value in enumerate(store[attr]):
            if (value == val):
                if (l[ind] == "yes"):
                    count1 += 1
                elif (l[ind] == "no"):
                    count2 += 1
            t = count1 + count2
        if t != 0:
            info_A[attr] += (t / rows) * (
                        (-1 * (count1 / t) * log(count1 / t, 2)) + (-1 * (count2 / t) * log(count2 / t, 2)))
            gain[attr] = info_D - info_A[attr]
print(info_A)
print(gain)
