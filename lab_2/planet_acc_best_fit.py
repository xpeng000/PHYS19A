import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

# import files and load them in 2D matrix (2xn): row[0] ==> time, row[1] ==> position
file_1 = np.loadtxt("data_1.txt", delimiter=",").transpose()
file_2 = np.loadtxt("data_2.txt", delimiter=",").transpose()
file_3 = np.loadtxt("data_3.txt", delimiter=",").transpose()
file_4 = np.loadtxt("data_4.txt", delimiter=",").transpose()
file_5 = np.loadtxt("data_5.txt", delimiter=",").transpose()

# position = init_position + 0.5*acceleration*time^2
# acceleration = 2*(position-init_position)/time^2

# calculate acceleration for each file
def find_acceleration(matrix): 
    numcols = len(matrix[0])
    print(numcols)
    time = matrix[0][numcols-1]-matrix[0][0]
    print(time)
    print(matrix[1][numcols-1])
    return 2*(matrix[1][numcols-1])/(time**2)

print("The acceleration of the planet in data_1.txt is ", find_acceleration(file_1))
print("The acceleration of the planet in data_2.txt is ", find_acceleration(file_2))
print("The acceleration of the planet in data_3.txt is ", find_acceleration(file_3))
print("The acceleration of the planet in data_4.txt is ", find_acceleration(file_4))
print("The acceleration of the planet in data_5.txt is ", find_acceleration(file_5))


# Method 1: Using Least Squares
# question 2
plt.plot(file_1[0], file_1[1], 'o')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position (raw data points)")
plt.show()

# question 3
def func_posi(t, a):
    return 0.5*a*t**2

par, cov = curve_fit(func_posi, file_1[0], file_1[1])
g_1 = par[0]
g_err = np.sqrt(cov[0, 0])
plt.plot(file_1[0], file_1[1], 'o')
plt.plot(file_1[0], func_posi(file_1[0], g_1))
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position with least squares curve fitting")
plt.show()

print("The acceleration obtained from performing least squares curve fitting is =  %.2f +/- %.2f m/s^2" %(g_1, g_err))
print()

# Method 2: Fitting with Linear Regression
deltaT = file_1[0][1]- file_1[0][0]
time = file_1[0]
position = file_1[1]
r, c = 2, len(file_1[1])-2  # velocity 2D array doesn't have velocity at time 0
velocity = r*[c*[0]]
velocity[0] = file_1[0]

for i in range(1, len(time)-1):
    velocity[0][i-1] = time[i]
for i in range(1, len(position)-1):
    velocity[1][i-1] = (position[i+1]-position[i-1])/(2*deltaT)

# print the first 10 velocity with its correponding time
for i in range(10):
    print("The velocity at time %.3f seconds is %.3f m/s" %(velocity[0][i], velocity[1][i]))
print()

# from Sagar's slide
velVsTime = np.empty((2, 0), float)
for i in range(1, len(time)-1):
    velocity = (file_1[1][i+1]-file_1[1][i-1])/(2*deltaT)
    velVsTime = np.append(velVsTime, [[file_1[0][i]], [velocity]], axis = 1)
for i in range(10):
    print("The velocity at time %.3f seconds is %.3f m/s" %(velVsTime[0][i], velVsTime[1][i]))


# use Scipy fit to find acceleration and its error
def func_velo(t, v0, a):
    return v0+a*t

par2, cov2 = curve_fit(func_velo, velVsTime[0], velVsTime[1])
v0 = par2[0]
g_2 = par2[1]
g_err_2 = np.sqrt(cov2[1, 1])
print(g_2)
print("The acceleration obtained from performing least squares curve fitting is =  %.2f +/- %.2f m/s^2" %(g_2, g_err_2))
print()

slope, intercept, r_value, p_value, std_err = stats.linregress(velVsTime[0], velVsTime[1])
plt.plot(velVsTime[0], velVsTime[1], 'o')
plt.plot(velVsTime[0], func_velo(velVsTime[0], g_2, v0))
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot','Best fit line g parameter: (%.3f +/- %.3f) m/s^2' %(slope, std_err)])
plt.title("Time vs Velocity with Linear Regression")
plt.show()


# Method 3: Numerical Differentiation
accVsTime = np.empty((2, 0), float)
for i in range(1, len(velVsTime[0])-1):
    acc = (velVsTime[1][i+1]-velVsTime[1][i-1])/(2*deltaT)
    accVsTime = np.append(accVsTime, [[velVsTime[0][i]], [acc]], axis = 1)
for i in range(10):
    print("The acceleration at time %.3f seconds is %.3f m/s" %(accVsTime[0][i], accVsTime[1][i]))

plt.plot(accVsTime[0], accVsTime[1])
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Time vs Acceleration with Numerical Differentiation")
plt.show()





