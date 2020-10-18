#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:42:40 2020

@author: xrachelpeng
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm

# import data into arrays 
down_raw = np.loadtxt("19a-InclinePlane-downward.txt", delimiter= ","). transpose()
up_raw = np.loadtxt("19a-InclinePlane-upward.txt", delimiter= ","). transpose()

# average position data
down = (down_raw[1]+down_raw[2]+down_raw[3]+down_raw[4]+down_raw[5])/100/5
up = (up_raw[1]+up_raw[2]+up_raw[3]+up_raw[4]+up_raw[5])/100/5


# method 1
# get velocity data through numerical analysis 
# downawards motion
velTime = np.empty((2, 0), float)
for i in range(1,len(down)-1):
    velocity = (down[i+1]-down[i-1])/(down_raw[0][i+1]-down_raw[0][i-1])
    velTime = np.append(velTime,[[down_raw[0,i]],[velocity]], axis=1)

# display velocity vs. time data 
print(velTime)

# linear fit (linregress) the velocity data to get accelerations
slope, intercept, r_value, p_value, std_err = stats.linregress(velTime[0], velTime[1])
# slope ==> acceleration 
print("The acceleration from linear regression is (%.3f +/- %.3f) m/s^2" %(slope, std_err))
print("The linear regression best fit function for velocity is V = %.3ft%.3f" %(slope, intercept))

# raw data plot
plt.plot(velTime[0], velTime[1], 'o')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot'])
plt.title("Time vs. Velocity data for downwards motion")
plt.show()

# plot velocity v. time data from best fit 
plt.plot(velTime[0], velTime[1], 'o')
plt.plot(velTime[0], velTime[0]*slope+intercept)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot','Linear regression V(t) = (%.3f*t + %.3f) m/s' %(slope, intercept)])
plt.title("Time vs. Velocity with linear regression for downwards motion")
plt.show()

# upwards motion
velTimeU = np.empty((2, 0), float)
for i in range(1,len(down)-1):
    velocity = (up[i+1]-up[i-1])/(up_raw[0][i+1]-up_raw[0][i-1])
    velTimeU = np.append(velTimeU,[[up_raw[0,i]],[velocity]], axis=1)

# display velocity vs. time data 
print(velTimeU)

# linear fit (linregress) the velocity data to get accelerations
slope, intercept, r_value, p_value, std_err = stats.linregress(velTimeU[0], velTimeU[1])
# slope ==> acceleration 
print("The acceleration from linear regression is (%.3f +/- %.3f) m/s^2" %(slope, std_err))
print("The linear regression best fit function for velocity is V = %.3ft + %.3f" %(slope, intercept))

# raw data plot
plt.plot(velTimeU[0], velTimeU[1], 'o')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot'])
plt.title("Time vs. Velocity data for upwards motion")
plt.show()

# plot velocity v. time data from best fit 
plt.plot(velTimeU[0], velTimeU[1], 'o')
plt.plot(velTimeU[0], velTimeU[0]*slope+intercept)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(['Scatter plot','Linear regression V(t) = (%.3f*t + %.3f) m/s' %(slope, intercept)])
plt.title("Time vs. Velocity with linear regression for upwards motion")
plt.show()

# method 2
# plot raw data 
time = down_raw[0]
plt.plot(time, down.transpose(), 'o')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend(['Scatter plot'])
plt.title("Time vs. Position data for downwards motion")
plt.show()

timeU = up_raw[0]
plt.plot(timeU, up.transpose(), 'o')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend(['Scatter plot'])
plt.title("Time vs. Position data for upwards motion")
plt.show()

# quadratic fitting
# downwards
def func(t, a, u, xo):
    return xo + u*t + 0.5*a*t**2
par, cov = curve_fit(func, time, down)
plt.plot(time, down.transpose(), 'o')
plt.plot(time, par[2]+par[1]*time+par[0]*0.5*time**2)
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend(['Scatter plot','Quadratic fitting P(t) = (%.3f*t^2 + %.3f*v0 + %.3f) m' %(par[0]*0.5, par[1], par[2])])
plt.title("Time vs. Position with quadratic fitting for downwards motion")
plt.show()
print("acc downwards: %.3f +/- %.3f" %(par[0], np.sqrt(cov[0, 0])))

# upwards
par2, cov2 = curve_fit(func, timeU, up)
plt.plot(timeU, up.transpose(), 'o')
plt.plot(timeU, par2[2]+par2[1]*timeU+par2[0]*0.5*timeU**2)
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend(['Scatter plot','Quadratic fitting P(t) = (%.3f*t^2 + %.3f*v0 + %.3f) m' %(par2[0]*0.5, par2[1], par2[2])])
plt.title("Time vs. Position with quadratic fitting for upwards motion")
plt.show()
print("acc upwards: %.3f +/- %.3f" %(par2[0], np.sqrt(cov2[0, 0])))

# plot residuals downards 
res = down.transpose()-func(time, par[0], par[1], par[2])
n, bins, patches = plt.hist(res, bins = 8)
mean = np.mean(res)
std = np.std(res)
xmin, xmax = plt.xlim()
x = np.linspace(xmin,xmax, 1000)
gaus = (bins[1]- bins[0])*len(res)*norm.pdf(x, mean,std)
plt.plot(x, gaus, 'k')
plt.title("Residual distribution for downwards motion mu = %.3f std=%.3f" %(mean, std))
plt.show()

# plot residuals upwards
resU = up.transpose()-func(timeU, par2[0], par2[1], par2[2])
n, bins, patches = plt.hist(resU, bins = 8)
mean = np.mean(resU)
std = np.std(resU)
xmin, xmax = plt.xlim()
x = np.linspace(xmin,xmax, 1000)
gaus = (bins[1]- bins[0])*len(resU)*norm.pdf(x, mean,std)
plt.plot(x, gaus, 'k')
plt.title("Residual distribution for upwards motion mu = %.3f std=%.3f" %(mean, std))
plt.show()

# combine res and resU to a new array
resNet = np.concatenate((res, resU))
n, bins, patches = plt.hist(resNet, bins = 8)
mean = np.mean(resNet)
std = np.std(resNet)
xmin, xmax = plt.xlim()
x = np.linspace(xmin,xmax, 1000)
gaus = (bins[1]- bins[0])*len(resNet)*norm.pdf(x, mean,std)
plt.plot(x, gaus, 'k')
plt.title("Residual distribution for upwards and downwards motion combined together mu = %.3f std=%.3f" %(mean, std))
plt.show()


# Chi_square
# downwards
error = 0.5/1000
chi = np.sum(((down.transpose() - func(time, par[0], par[1], par[2]))/error)**2)
ndof = len(time) - 3
chi_square = chi/ndof
print(chi_square)

# upwards
chiU = np.sum(((up.transpose() - func(timeU, par2[0], par2[1], par2[2]))/error)**2)
ndofU = len(timeU) - 3
chi_squareU = chiU/ndofU 
print(chi_squareU)




