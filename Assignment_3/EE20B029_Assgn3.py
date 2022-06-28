#             Assignment 3
#             A3: Fitting Data to ModelsAssignment
#             Name: C. Dheeraj Sai
#             Roll No.: EE20B029


import scipy.special as sp
from scipy.linalg import lstsq
from pylab import *


# Defining the original function g seperately
def g(t, A, B):
    return A * sp.jn(2, t) + B * t


# Function for getting the indexes of an element defined in a 2x2 matrix
def find(element, matrix):
    for i, matrix_i in enumerate(matrix):
        for j, value in enumerate(matrix_i):
            if value == element:
                return (i, j)


#####
fittingMatrix = loadtxt("Assignment_3/fitting.dat", dtype=float)
N = len(fittingMatrix)  #                  No. of time instances
k = len(fittingMatrix[0]) - 1  #           No. of signals
time = fittingMatrix[:, 0]  #              time array
data = fittingMatrix[:, 1:]  #             data matrix
sigma = logspace(-1, -3, k)  #             sigma array which was used to add noise
####

# \u03c3 and \u2192 are the unicode values of sigma and right arroe charectors respectively

# Plot of the ideal curve along with plots of the function with noise added
figure()
for i in range(k):
    plot(time, data[:, i], label=f"\u03c3={sigma[i]:.3f}")
plot(time, g(time, 1.05, -0.105), "k-", label="True Value", linewidth=2)
title("Q4: Data to be fitted to theory\n", fontsize=16, fontweight="bold")
legend(loc="upper right")
xlabel("t \u2192", fontsize=12)
ylabel("f(t) + noises \u2192", fontsize=12)
grid()
####


first_col = data[:, 0]

# Plot of the ideal curve along with the deviations of data corresponding to sigma = 0.1
figure()
errorbar(time[::5], first_col[::5], sigma[0], fmt="ro", label="Errorbar")
plot(time, g(time, 1.05, -0.105), "k", label="f(t)")
title(
    "Q5: Data points for \u03c3 = 0.1 along with exact function\n",
    fontsize=16,
    fontweight="bold",
)
legend(loc="upper right")
xlabel("t \u2192", fontsize=12)
grid()
####

# Checking if the values of function obtained directly and obtained via matrix multiplication are equal
M = c_[sp.jn(2, time), time]
A0 = 1.05
B0 = -0.105
p = array([A0, B0]).reshape(2, 1)

g1vector = M @ p
g2vector = array(g(time, A0, B0)).reshape(N, 1)

if allclose(g1vector, g2vector):
    print("Vectors g1 and g2 are equal\n")
else:
    print("Vectors g1 and g2 are not equal")
####


# Finding the mean sqared error between the ideal curve and the signal corresponding to the first coloumn
A = linspace(0, 2, 21)
B = linspace(-0.2, 0, 21)
e = zeros((21, 21))
for i in range(21):
    for j in range(21):
        for n in range(N):
            e[i][j] += ((data[:, 0][n] - g(time[n], A[i], B[j])) ** 2) / 101


# Generating a contour plot of the mean squared error corresponding to the signal in the first coloumn
figure()
plot3 = contour(A, B, e, 20)
title("Q8: Contour plot of ∈ij\n", fontsize=16, fontweight="bold")
xlabel("A \u2192", fontsize=12)
ylabel("B \u2192", fontsize=12)
clabel(plot3, inline=1, fontsize=10)
minValue = e.min()
index = find(minValue, e)
plot(A[index[0]], B[index[1]], "ro", ms=12)
annotate("Exact Location", xy=(A[index[0]], B[index[1]]))
###


# Obtaining the best estimate of the parameters A and B using lstsq function and findint the absolute error in each signal
coeff = asarray([lstsq(M, data[:, i], rcond=None)[0] for i in range(k)])
errorA = abs(coeff[:, 0] - A0)
errorB = abs(coeff[:, 1] - B0)


# Plotting the variation of the absolute error in all the signals in a linear scale
figure()
plot(sigma, errorA, "ro--", label="Aerr")
plot(sigma, errorB, "go--", label="Berr")
title("Q10: Variation of error with noise\n", fontsize=16, fontweight="bold")
xlabel("Non standard deviation \u2192", fontsize=12)
ylabel("MS error \u2192", fontsize=12)
grid()
legend(loc="upper left")
####


# Plotting the variation of the absolute error in all the signals in a logarithmic scale
figure()
loglog(sigma, errorA, "ro", label="Aerr")
loglog(sigma, errorA, "go", label="Berr")
stem(sigma, errorA, linefmt="r-", markerfmt="ro")
stem(sigma, errorB, linefmt="g-", markerfmt="go")
title("Q10: Variation of error with noise\n", fontsize=16, fontweight="bold")
xlabel("\u03c3n \u2192", fontsize=12)
ylabel("MS error \u2192", fontsize=12)
legend(loc="upper right")
grid()
####

# Displaying all the plots to be generated
show()
