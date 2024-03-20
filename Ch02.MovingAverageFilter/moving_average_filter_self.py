import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

v = np.random.normal(0,1, 10 )
t = np.arange(0, 10, 1)

# print(v, t)

def get_volt():
    v = np.random.normal(0, 4)
    volt_mean = 14.4
    volt_meas = volt_mean + v
    return volt_meas

def average_filter(k, x_meas, x_avg):

    alpha = (k - 1) / k
    x_avg = alpha * x_avg + (1-alpha) * x_meas
    return x_avg

time = np.arange(0,1000,0.2)
n_sample = len(time)

x_meas_save = np.zeros(n_sample)
x_avg_save = np.zeros(n_sample)
x_avg = 0

for i in range (n_sample):

    k = i + 1
    x_meas = get_volt()
    x_meas_save[i] = x_meas
    x_avg = average_filter(k, x_meas, x_avg)
    x_avg_save[i] = x_avg

print(x_meas_save)
print(x_avg_save)


plt.plot(time, x_meas_save, 'r*-', label='Measured')
plt.plot(time, x_avg_save, 'b-', label='Average')
plt.legend(loc='upper left')
plt.title('Measured Voltages v.s. Average Filter Values')
plt.xlabel('Time [sec]')
plt.ylabel('Volt [V]')
plt.show()
plt.savefig('png/average_filter.png')