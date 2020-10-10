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

# Method 1: Using Least Squares
# question 2
plt.plot(file_1[0], file_1[1], 'o')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position for data_1.txt (raw data points)")
plt.show()

plt.plot(file_2[0], file_2[1], 'o')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position for data_2.txt (raw data points)")
plt.show()

plt.plot(file_3[0], file_3[1], 'o')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position for data_3.txt (raw data points)")
plt.show()

plt.plot(file_4[0], file_4[1], 'o')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position for data_4.txt (raw data points)")
plt.show()

plt.plot(file_5[0], file_5[1], 'o')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position for data_5.txt (raw data points)")
plt.show()


# question 3
def func_posi(t, a):
    return 0.5*a*t**2

par, cov = curve_fit(func_posi, file_1[0], file_1[1])
g_1_1 = par[0]
g_err_1 = np.sqrt(cov[0, 0])
plt.plot(file_1[0], file_1[1], 'o')
plt.plot(file_1[0], func_posi(file_1[0], g_1_1))
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position with least squares curve fitting for data_1.txt")
plt.show()

print("The acceleration obtained from performing least squares curve fitting is =  %.2f +/- %.2f m/s^2" %(g_1_1, g_err_1))
print()

par2, cov2 = curve_fit(func_posi, file_2[0], file_2[1])
g_1_2 = par2[0]
g_err_2 = np.sqrt(cov2[0, 0])
plt.plot(file_2[0], file_2[1], 'o')
plt.plot(file_2[0], func_posi(file_2[0], g_1_2))
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position with least squares curve fitting for data_2.txt")
plt.show()

print("The acceleration obtained from performing least squares curve fitting is =  %.2f +/- %.2f m/s^2" %(g_1_2, g_err_2))
print()

par3, cov3 = curve_fit(func_posi, file_3[0], file_3[1])
g_1_3 = par3[0]
g_err_3 = np.sqrt(cov3[0, 0])
plt.plot(file_3[0], file_3[1], 'o')
plt.plot(file_3[0], func_posi(file_3[0], g_1_3))
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position with least squares curve fitting for data_3.txt")
plt.show()

print("The acceleration obtained from performing least squares curve fitting is =  %.2f +/- %.2f m/s^2" %(g_1_3, g_err_3))
print()

par4, cov4 = curve_fit(func_posi, file_4[0], file_4[1])
g_1_4 = par4[0]
g_err_4 = np.sqrt(cov4[0, 0])
plt.plot(file_4[0], file_4[1], 'o')
plt.plot(file_4[0], func_posi(file_4[0], g_1_4))
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position with least squares curve fitting for data_4.txt")
plt.show()

print("The acceleration obtained from performing least squares curve fitting is =  %.2f +/- %.2f m/s^2" %(g_1_4, g_err_4))
print()

par5, cov5 = curve_fit(func_posi, file_5[0], file_5[1])
g_1_5 = par5[0]
g_err_5 = np.sqrt(cov4[0, 0])
plt.plot(file_5[0], file_5[1], 'o')
plt.plot(file_5[0], func_posi(file_5[0], g_1_5))
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Time vs Position with least squares curve fitting for data_5.txt")
plt.show()

print()
print("The acceleration obtained from performing least squares curve fitting for data_1.txt is =  %.2f +/- %.2f m/s^2" %(g_1_1, g_err_1))
print("The acceleration obtained from performing least squares curve fitting for data_2.txt is =  %.2f +/- %.2f m/s^2" %(g_1_2, g_err_2))
print("The acceleration obtained from performing least squares curve fitting for data_3.txt is =  %.2f +/- %.2f m/s^2" %(g_1_3, g_err_3))
print("The acceleration obtained from performing least squares curve fitting for data_4.txt is =  %.2f +/- %.2f m/s^2" %(g_1_4, g_err_4))
print("The acceleration obtained from performing least squares curve fitting for data_5.txt is =  %.2f +/- %.2f m/s^2" %(g_1_5, g_err_5))
print()

# acceleration weighted average:
weighted_ave = (g_1_1/g_err_1**2+g_1_2/g_err_2**2+g_1_3/g_err_3**2+g_1_4/g_err_4**2+g_1_5/g_err_5**2)/(1/g_err_1**2+1/g_err_2**2+1/g_err_3**2+1/g_err_4**2+1/g_err_5**2)
weighted_err = np.sqrt(1/(1/g_err_1**2+1/g_err_2**2+1/g_err_3**2+1/g_err_4**2+1/g_err_5**2))
print("The weighted average acceleration is %.2f +/- %.2f m/s^2" %(weighted_ave, weighted_err))
        

# Method 2: Fitting with Linear Regression
deltaT = file_1[0][1]- file_1[0][0]

# question 6, from Sagar's slide
velVsTime_1 = np.empty((2, 0), float)
for i in range(1, len(file_1[0])-1):
    velocity_1 = (file_1[1][i+1]-file_1[1][i-1])/(2*deltaT)
    velVsTime_1 = np.append(velVsTime_1, [[file_1[0][i]], [velocity_1]], axis = 1)
print("Data_1.txt file:")
for i in range(10):
    print("The velocity at time %.3f seconds is %.3f m/s" %(velVsTime_1[0][i], velVsTime_1[1][i]))

velVsTime_2 = np.empty((2, 0), float)
for i in range(1, len(file_2[0])-1):
    velocity_2 = (file_2[1][i+1]-file_2[1][i-1])/(2*deltaT)
    velVsTime_2 = np.append(velVsTime_2, [[file_2[0][i]], [velocity_2]], axis = 1)
print("Data_2.txt file:")
for i in range(10):
    print("The velocity at time %.3f seconds is %.3f m/s" %(velVsTime_2[0][i], velVsTime_2[1][i]))
    
velVsTime_3 = np.empty((2, 0), float)
for i in range(1, len(file_3[0])-1):
    velocity_3 = (file_3[1][i+1]-file_3[1][i-1])/(2*deltaT)
    velVsTime_3 = np.append(velVsTime_3, [[file_3[0][i]], [velocity_3]], axis = 1)
print("Data_3.txt file:")
for i in range(10):
    print("The velocity at time %.3f seconds is %.3f m/s" %(velVsTime_3[0][i], velVsTime_3[1][i]))
    
velVsTime_4 = np.empty((2, 0), float)
for i in range(1, len(file_4[0])-1):
    velocity_4 = (file_4[1][i+1]-file_4[1][i-1])/(2*deltaT)
    velVsTime_4 = np.append(velVsTime_4, [[file_4[0][i]], [velocity_4]], axis = 1)
print("Data_4.txt file:")
for i in range(10):
    print("The velocity at time %.3f seconds is %.3f m/s" %(velVsTime_4[0][i], velVsTime_4[1][i]))
    
velVsTime_5 = np.empty((2, 0), float)
for i in range(1, len(file_5[0])-1):
    velocity_5 = (file_5[1][i+1]-file_5[1][i-1])/(2*deltaT)
    velVsTime_5 = np.append(velVsTime_5, [[file_5[0][i]], [velocity_5]], axis = 1)
print("Data_5.txt file:")
for i in range(10):
    print("The velocity at time %.3f seconds is %.3f m/s" %(velVsTime_5[0][i], velVsTime_5[1][i]))

# use Scipy fit to find acceleration and its error
def func_velo(t, a):
    return a*t

# question 7
par21, cov21 = curve_fit(func_velo, velVsTime_1[0], velVsTime_1[1])
g_21 = par21[0]
g_err_21 = np.sqrt(cov21[0, 0])
print("The acceleration obtained from performing least squares curve fitting for data_1.txt is =  %.2f +/- %.2f m/s^2" %(g_21, g_err_21))

par22, cov22 = curve_fit(func_velo, velVsTime_2[0], velVsTime_2[1])
g_22 = par22[0]
g_err_22 = np.sqrt(cov22[0, 0])
print("The acceleration obtained from performing least squares curve fitting for data_2.txt is =  %.2f +/- %.2f m/s^2" %(g_22, g_err_22))


par23, cov23 = curve_fit(func_velo, velVsTime_3[0], velVsTime_3[1])
g_23 = par23[0]
g_err_23 = np.sqrt(cov23[0, 0])
print("The acceleration obtained from performing least squares curve fitting for data_3.txt is =  %.2f +/- %.2f m/s^2" %(g_23, g_err_23))


par24, cov24 = curve_fit(func_velo, velVsTime_4[0], velVsTime_4[1])
g_24 = par24[0]
g_err_24 = np.sqrt(cov24[0, 0])
print("The acceleration obtained from performing least squares curve fitting for data_4.txt is =  %.2f +/- %.2f m/s^2" %(g_24, g_err_24))


par25, cov25 = curve_fit(func_velo, velVsTime_5[0], velVsTime_5[1])
g_25 = par25[0]
g_err_25 = np.sqrt(cov25[0, 0])
print("The acceleration obtained from performing least squares curve fitting for data_5.txt is =  %.2f +/- %.2f m/s^2" %(g_25, g_err_25))
print()

# acceleration weighted average:
weighted_ave_2 = (g_21/g_err_21**2+g_22/g_err_22**2+g_23/g_err_23**2+g_24/g_err_24**2+g_25/g_err_25**2)/(1/g_err_21**2+1/g_err_22**2+1/g_err_23**2+1/g_err_24**2+1/g_err_25**2)
weighted_err_2 = np.sqrt(1/(1/g_err_21**2+1/g_err_22**2+1/g_err_23**2+1/g_err_24**2+1/g_err_25**2))
print("The weighted average acceleration is %.2f +/- %.2f m/s^2" %(weighted_ave_2, weighted_err_2))

# question 8
slope, intercept, r_value, p_value, std_err = stats.linregress(velVsTime_1[0], velVsTime_1[1])
plt.plot(velVsTime_1[0], velVsTime_1[1], 'o')
plt.plot(velVsTime_1[0], func_velo(velVsTime_1[0], g_21))
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot','Best fit line g parameter: (%.2f +/- %.2f) m/s^2' %(slope, std_err)])
plt.title("Time vs Velocity with Linear Regression from data_1.txt")
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(velVsTime_2[0], velVsTime_2[1])
plt.plot(velVsTime_2[0], velVsTime_2[1], 'o')
plt.plot(velVsTime_2[0], func_velo(velVsTime_2[0], g_22))
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot','Best fit line g parameter: (%.2f +/- %.2f) m/s^2' %(slope, std_err)])
plt.title("Time vs Velocity with Linear Regression from data_2.txt")
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(velVsTime_3[0], velVsTime_3[1])
plt.plot(velVsTime_3[0], velVsTime_3[1], 'o')
plt.plot(velVsTime_3[0], func_velo(velVsTime_3[0], g_23))
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot','Best fit line g parameter: (%.2f +/- %.2f) m/s^2' %(slope, std_err)])
plt.title("Time vs Velocity with Linear Regression from data_3.txt")
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(velVsTime_4[0], velVsTime_4[1])
plt.plot(velVsTime_4[0], velVsTime_4[1], 'o')
plt.plot(velVsTime_4[0], func_velo(velVsTime_4[0], g_24))
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot','Best fit line g parameter: (%.2f +/- %.2f) m/s^2' %(slope, std_err)])
plt.title("Time vs Velocity with Linear Regression from data_4.txt")
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(velVsTime_5[0], velVsTime_5[1])
plt.plot(velVsTime_5[0], velVsTime_5[1], 'o')
plt.plot(velVsTime_5[0], func_velo(velVsTime_5[0], g_25))
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot','Best fit line g parameter: (%.2f +/- %.2f) m/s^2' %(slope, std_err)])
plt.title("Time vs Velocity with Linear Regression from data_5.txt")
plt.show()


# Method 3: Numerical Differentiation
# question 9
accVsTime = np.empty((2, 0), float)
for i in range(1, len(velVsTime_1[0])-1):
    acc = (velVsTime_1[1][i+1]-velVsTime_1[1][i-1])/(2*deltaT)
    accVsTime = np.append(accVsTime, [[velVsTime_1[0][i]], [acc]], axis = 1)
print("First 10 accelerations with time from data_1.txt")
for i in range(10):
    print("The acceleration at time %.3f seconds is %.3f m/s^2" %(accVsTime[0][i], accVsTime[1][i]))
print()

plt.plot(accVsTime[0], accVsTime[1])
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Time vs Acceleration with Numerical Differentiation from data_1.txt")
plt.show()

accVsTime_2 = np.empty((2, 0), float)
for i in range(1, len(velVsTime_2[0])-1):
    acc = (velVsTime_2[1][i+1]-velVsTime_2[1][i-1])/(2*deltaT)
    accVsTime_2 = np.append(accVsTime_2, [[velVsTime_2[0][i]], [acc]], axis = 1)
print("First 10 accelerations with time from data_2.txt")
for i in range(10):
    print("The acceleration at time %.3f seconds is %.3f m/s^2" %(accVsTime_2[0][i], accVsTime_2[1][i]))
print()

plt.plot(accVsTime_2[0], accVsTime_2[1])
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Time vs Acceleration with Numerical Differentiation from data_2.txt")
plt.show()


accVsTime_3 = np.empty((2, 0), float)
for i in range(1, len(velVsTime_3[0])-1):
    acc = (velVsTime_3[1][i+1]-velVsTime_3[1][i-1])/(2*deltaT)
    accVsTime_3 = np.append(accVsTime_3, [[velVsTime_3[0][i]], [acc]], axis = 1)
print("First 10 accelerations with time from data_3.txt")
for i in range(10):
    print("The acceleration at time %.3f seconds is %.3f m/s^2" %(accVsTime_3[0][i], accVsTime_3[1][i]))
print()

plt.plot(accVsTime_3[0], accVsTime_3[1])
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Time vs Acceleration with Numerical Differentiation from data_3.txt")
plt.show()

accVsTime_4 = np.empty((2, 0), float)
for i in range(1, len(velVsTime_4[0])-1):
    acc = (velVsTime_4[1][i+1]-velVsTime_4[1][i-1])/(2*deltaT)
    accVsTime_4 = np.append(accVsTime_4, [[velVsTime_4[0][i]], [acc]], axis = 1)
print("First 10 accelerations with time from data_4.txt")
for i in range(10):
    print("The acceleration at time %.3f seconds is %.3f m/s^2" %(accVsTime_4[0][i], accVsTime_4[1][i]))
print()

plt.plot(accVsTime_4[0], accVsTime_4[1])
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Time vs Acceleration with Numerical Differentiation from data_4.txt")
plt.show()

accVsTime_5 = np.empty((2, 0), float)
for i in range(1, len(velVsTime_5[0])-1):
    acc = (velVsTime_5[1][i+1]-velVsTime_5[1][i-1])/(2*deltaT)
    accVsTime_5 = np.append(accVsTime_5, [[velVsTime_5[0][i]], [acc]], axis = 1)
print("First 10 accelerations with time from data_5.txt")
for i in range(10):
    print("The acceleration at time %.3f seconds is %.3f m/s^2" %(accVsTime_5[0][i], accVsTime_5[1][i]))
print()

plt.plot(accVsTime_5[0], accVsTime_5[1])
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Time vs Acceleration with Numerical Differentiation from data_5.txt")
plt.show()

# question 10
err1 = np.sqrt(np.var(accVsTime[1]))/np.sqrt(len(accVsTime[1]))
err2 = np.sqrt(np.var(accVsTime_2[1]))/np.sqrt(len(accVsTime_2[1]))
err3 = np.sqrt(np.var(accVsTime_3[1]))/np.sqrt(len(accVsTime_3[1]))
err4 = np.sqrt(np.var(accVsTime_4[1]))/np.sqrt(len(accVsTime_4[1]))
err5 = np.sqrt(np.var(accVsTime_5[1]))/np.sqrt(len(accVsTime_5[1]))

mean1 = np.mean(accVsTime[1])
mean2 = np.mean(accVsTime_2[1])
mean3 = np.mean(accVsTime_3[1])
mean4 = np.mean(accVsTime_4[1])
mean5 = np.mean(accVsTime_5[1])

print("The average acceleration in data_1.txt using numerical differentiation is (%.2f +/- %.2f) m/s^2" %(mean1, err1))
print("The average acceleration in data_2.txt using numerical differentiation is (%.2f +/- %.2f) m/s^2" %(mean2, err2))
print("The average acceleration in data_3.txt using numerical differentiation is (%.2f +/- %.2f) m/s^2" %(mean3, err3))
print("The average acceleration in data_4.txt using numerical differentiation is (%.2f +/- %.2f) m/s^2" %(mean4, err4))
print("The average acceleration in data_5.txt using numerical differentiation is (%.2f +/- %.2f) m/s^2" %(mean5, err5))

# acceleration weighted average:
weighted_ave_3 = (mean1/err1**2+mean2/err2**2+mean3/err3**2+mean4/err4**2+mean5/err5**2)/(1/(1/err1**2+1/err2**2+1/err3**2+1/err4**2+1/err5**2))
weighted_err_3 = np.sqrt(1/(1/err1**2+1/err2**2+1/err3**2+1/err4**2+1/err5**2))
print("The weighted average acceleration is %.2f +/- %.2f m/s^2" %(weighted_ave_3, weighted_err_3))

print(accVsTime_5)

# question 11
accInt = []
slope, intercept, r_value, p_value, std_err = stats.linregress(accVsTime[0], accVsTime[1])
accInt.append(intercept)
plt.plot(accVsTime[0], accVsTime[1], 'o', label="Scatter plot")
plt.plot(accVsTime[0], intercept+slope*accVsTime[0], label = "Best fit line: acceleration = %.2f t + %.2f " %(slope, intercept))
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend(['Scatter plot', "Best fit line: acceleration = %.2f t + %.2f m/s^2" %(slope, intercept)])
plt.title("Acceleration vs Time with Linear Regression from data_1.txt")
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(accVsTime_2[0], accVsTime_2[1])
accInt.append(intercept)
plt.plot(accVsTime_2[0], accVsTime_2[1], 'o', label="Scatter plot")
plt.plot(accVsTime_2[0], intercept+slope*accVsTime_2[0], label = "Best fit line: acceleration = %.2f t + %.2f " %(slope, intercept))
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend(['Scatter plot', "Best fit line: acceleration = %.2f t + %.2f m/s^2" %(slope, intercept)])
plt.title("Acceleration vs Time with Linear Regression from data_2.txt")
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(accVsTime_3[0], accVsTime_3[1])
accInt.append(intercept)
plt.plot(accVsTime_3[0], accVsTime_3[1], 'o', label="Scatter plot")
plt.plot(accVsTime_3[0], intercept+slope*accVsTime_3[0], label = "Best fit line: acceleration = %.2f t + %.2f " %(slope, intercept))
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend(['Scatter plot', "Best fit line: acceleration = %.2f t + %.2f m/s^2" %(slope, intercept)])
plt.title("Acceleration vs Time with Linear Regression from data_3.txt")
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(accVsTime_4[0], accVsTime_4[1])
accInt.append(intercept)
plt.plot(accVsTime_4[0], accVsTime_4[1], 'o', label="Scatter plot")
plt.plot(accVsTime_4[0], intercept+slope*accVsTime_4[0], label = "Best fit line: acceleration = %.2f t + %.2f " %(slope, intercept))
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend(['Scatter plot', "Best fit line: acceleration = %.2f t + %.2f m/s^2" %(slope, intercept)])
plt.title("Acceleration vs Time with Linear Regression from data_4.txt")
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(accVsTime_5[0], accVsTime_5[1])
accInt.append(intercept)
plt.plot(accVsTime_5[0], accVsTime_5[1], 'o', label="Scatter plot")
plt.plot(accVsTime_5[0], intercept+slope*accVsTime_5[0], label = "Best fit line: acceleration = %.2f t + %.2f " %(slope, intercept))
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend(['Scatter plot', "Best fit line: acceleration = %.2f t + %.2f m/s^2" %(slope, intercept)])
plt.title("Acceleration vs Time with Linear Regression from data_5.txt")
plt.show()

# calculate ave acceleration and its error
accMean = np.mean(accInt)
accStd = np.std(accInt)
accErr = accStd/(5**0.5)
print("The average acceleration is %.2f +/- %.2f m/s^2" %(accMean, accErr))
















