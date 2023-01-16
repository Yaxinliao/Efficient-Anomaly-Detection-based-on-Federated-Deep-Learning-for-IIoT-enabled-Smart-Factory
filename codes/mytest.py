import numpy as np
import random

random.seed(4)
a = random.sample(range(2000, 70000), 20)
a.sort()
print(a)
for i in range(1, 20):
    print(a[i] - a[i-1])

