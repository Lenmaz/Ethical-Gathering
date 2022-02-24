import numpy as np

Q = np.load("Q_func.npy")
V = np.load("V_func.npy")

new_Q = list()

len_states = len(Q)

for i in range(len_states):
    new_max = np.max(Q[i][:, 0])
    new_Q.append(new_max)

for i in range(len_states):
    print(new_Q[i], V[i])