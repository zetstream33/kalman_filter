import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# v = np.random.normal(0,1, 10 )
# t = np.arange(0, 10, 1)
#
# # print(v, t)
#
# def get_volt(x):
#     noise = np.random.normal(0, 1)
#     y = x + noise
#     return y
#
# def average_filter(k, x_meas, x_avg):
#
#     alpha = (k - 1) / k
#     x_avg = alpha * x_avg + (1-alpha) * x_meas
#     return x_avg
#
# def moving_average_filter_1step(x_mavg, x_meas):
#
#     x_mavg = (0.5 * x_mavg) + (0.5 * x_meas)
#     return x_mavg
#
#
# time = np.arange(0,10,0.2)
# n_sample = len(time)
#
# x_meas_save = np.zeros(n_sample)
# x_avg_save = np.zeros(n_sample)
# x_mavg_save= np.zeros(n_sample)
#
# x_avg = 0
# x_mavg = 0
#
# for i in range (n_sample):
#
#     k = i + 1
#     x_meas = get_volt(k)
#     x_meas_save[i] = x_meas
#
#     x_avg = average_filter(k, x_meas, x_avg)
#     x_avg_save[i] = x_avg
#
#     x_mavg = moving_average_filter_1step(x_mavg, x_meas)
#     x_mavg_save[i] = x_mavg
#
    

"""----------------------------------------------------"""

from scipy import io
input_mat = io.loadmat('../data/2.MovAvgFilter/SonarAlt.mat')
def get_sonar(i):
    """Measure sonar."""
    z = input_mat['sonarAlt'][0][i]  # input_mat['sonaralt']: (1, 1501)
    return z

def moving_average_filter_nstep(x_meas, x_n):
    n = len(x_n)
    np.set_printoptions(precision=2)
    print("x_n : {}".format(x_n))
    for i in range(n-1):
        x_n[i] = x_n[i + 1]
    
    x_n[n-1] = x_meas
    x_avg = np.mean(x_n)
    
    return x_avg, x_n

n = 10
n_samples = 1000
time_end = 10
dt = time_end / n_samples
time = np.arange(0, time_end, dt)
x_meas_save = np.zeros(n_samples)
x_avg_save = np.zeros(n_samples)

for i in range(n_samples):
    x_meas = get_sonar(i)

    if i == 0:
        x_avg, x_n = x_meas, x_meas * np.ones(n)
    else:
        x_avg, x_n = moving_average_filter_nstep(x_meas, x_n)

    x_meas_save[i] = x_meas
    x_avg_save[i] = x_avg

plt.plot(time, x_meas_save, 'r*', label='Measured')
plt.plot(time, x_avg_save, 'b-', label='Moving average')
plt.legend(loc='upper left')
plt.title('Measured Altitudes v.s. Moving Average Filter Values')
plt.xlabel('Time [sec]')
plt.ylabel('Altitude [m]')
plt.savefig('png/moving_average_filter_self.png')
plt.show()





#
#
# plt.plot(time, x_meas_save, 'r*-', label='Measured')
# plt.plot(time, x_avg_save, 'b-', label='Average Filter')
# plt.plot(time, x_mavg_save, 'g-', label='Moving Average Filter (1 Step Back) ')
#
#
#
#
# plt.legend(loc='upper left')
# plt.title('Measured Voltages v.s. Average Filter Values')
# plt.xlabel('Time [sec]')
# plt.ylabel('Volt [V]')
# plt.show()
# plt.savefig('png/movig_average_filter & average_filter.png')