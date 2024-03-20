"""

구현된 Low_Pass_Filter Reference 는 Moving이 구현돼 있지 않음!
이에 코드를 직접 작성해봄.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

input_mat = io.loadmat('../data/3.LPF/SonarAlt.mat')
def get_sonar(i):
    """Measure sonar."""
    z = input_mat['sonarAlt'][0][i]  # input_mat['sonaralt']: (1, 1501)
    return z

def low_pass_filter(x_meas, x_n, weight):

    n = len(x_n)

    for i in range(n - 1): # 길이 : n-1
        x_n[i] = x_n[i+1]

    x_n[n-1] = x_meas
    x_avg = np.mean(x_n)

    x_esti = (1-weight) * x_avg + weight * x_meas

    return x_esti



weight = 0.4
n = 10

n_sample = 500
time_end = 50
dt = time_end / n_sample

time = np.arange(0,time_end,dt)

x_meas_save = np.zeros(n_sample)
x_esti_save = np.zeros(n_sample)

x_esti = 0


for i in range(n_sample):

    x_meas = get_sonar(i)

    if i == 0:
        x_esti = x_meas
        x_n = x_meas * np.ones(n)
    else:
        x_esti = low_pass_filter(x_meas, x_n, weight)

    x_meas_save[i] = x_meas
    x_esti_save[i] = x_esti

plt.plot(time, x_meas_save, 'r*-', label='Measured')
plt.plot(time, x_esti_save, 'b-', label='Low Pass Filter')
plt.legend(loc='upper left')
plt.title('Measured Altitudes v.s. Low Pass Filter Values')
plt.xlabel('Time [sec]')
plt.ylabel('Altitude [m]')
plt.savefig('png/low_pass_filter_self.png')
plt.show()
