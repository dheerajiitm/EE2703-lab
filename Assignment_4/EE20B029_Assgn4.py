#             Assignment 4
#             A4: Fourier Approximations
#             Name: C. Dheeraj Sai
#             Roll No.: EE20B029


from pylab import *
from scipy.integrate import quad
from scipy.linalg import lstsq

# Functions given
def e(x):
    return exp(x)


def coscos(x):
    return cos(cos(x))


####

# Functions used in quad method
def u1(x, k):
    return exp(x) * cos(k * x)


def u2(x, k):
    return cos(cos(x)) * cos(k * x)


def v1(x, k):
    return exp(x) * sin(k * x)


def v2(x, k):
    return cos(cos(x)) * sin(k * x)


####


# No. of data points in x-axis
x = linspace(-2 * pi, 4 * pi, 1200)
period = linspace(0, 2 * pi, 400)
period = asarray(period.tolist() * 3)
####

# Plot of true value and the periodic extension
figure()
semilogy(x, e(x), "r", label="True Value")
semilogy(x, e(period), "b", label="Periodic Extension")
xlabel("x \u2192", fontsize=12)
ylabel("e^(x) \u2192", fontsize=12)
legend(loc="upper right")
grid()

# Plot of cos(cos(x))
figure()
plot(x, coscos(x), "b")
xlabel("x \u2192", fontsize=12)
ylabel("cos(cos(x)) \u2192", fontsize=12)
grid()

# Finding the fourier coefficients
coeff_exp = np.zeros(51)
coeff_coscos = np.zeros(51)
coeff_exp[0] = quad(e, 0, 2 * pi)[0] / (2 * pi)
coeff_coscos[0] = quad(coscos, 0, 2 * pi)[0] / (2 * pi)
for k in range(1, 51, 2):
    coeff_exp[k] = quad(u1, 0, 2 * pi, args=((k + 1) / 2))[0] / pi
    coeff_coscos[k] = quad(u2, 0, 2 * pi, args=((k + 1) / 2))[0] / pi
for k in range(2, 51, 2):
    coeff_exp[k] = quad(v1, 0, 2 * pi, args=(k / 2))[0] / pi
    coeff_coscos[k] = quad(v2, 0, 2 * pi, args=(k / 2))[0] / pi
####

##Semilog plot of fourier coefficients of e^(x)
figure()
title("Semilog plot of fourier coefficients of e^(x)", fontsize=16)
semilogy(range(51), abs(coeff_exp), "ro")
xlabel("n \u2192", fontsize=12)
ylabel("Magnitude of coefficients \u2192", fontsize=12)
grid()

##Loglog plot of fourier coefficients of e^(x)
figure()
title("Loglog plot of fourier coefficients of e^(x)", fontsize=16)
loglog(range(51), abs(coeff_exp), "ro")
xlabel("n \u2192", fontsize=12)
ylabel("Magnitude of coefficients \u2192", fontsize=12)
grid()

##Semilog plot of fourier coefficients of cos(cos(x))
figure()
title("Semilog plot of fourier coefficients of cos(cos(x))", fontsize=16)
semilogy(range(51), abs(coeff_coscos), "ro")
xlabel("n \u2192", fontsize=12)
ylabel("Magnitude of coefficients \u2192", fontsize=12)
grid()

##Loglog plot of fourier coefficients of cos(cos(x))
figure()
title("Loglog plot of fourier coefficients of e^(x)", fontsize=16)
loglog(range(51), abs(coeff_coscos), "ro")
xlabel("n \u2192", fontsize=12)
ylabel("Magnitude of coefficients \u2192", fontsize=12)
grid()


# Finding fourier coefficients using least squares method
x1 = linspace(0, 2 * pi, 401)
x1 = x1[:-1]
b1 = e(x1)
b2 = coscos(x1)
A = zeros((400, 51))
A[:, 0] = 1
for k in range(1, 26):
    A[:, 2 * k - 1] = cos(k * x1)
    A[:, 2 * k] = sin(k * x1)


c_exp = lstsq(A, b1)[0]
c_coscos = lstsq(A, b2)[0]
####

# Semilog plot of fourier coefficients of e^(x) using least Squares aproach
figure()
semilogy(range(51), abs(c_exp), "go", label="Least Squares Approach")
semilogy(range(51), abs(coeff_exp), "ro", label="True Value")
xlabel("n \u2192", fontsize=12)
ylabel("Fourier coefficients \u2192", fontsize=12)
legend(loc="upper right")
grid()

# Loglog plot of fourier coefficients of e^(x) using least Squares aproach
figure()
loglog(range(51), abs(c_exp), "go", label="Least Squares Approach")
loglog(range(51), abs(coeff_exp), "ro", label="True Value")
xlabel("n \u2192", fontsize=12)
ylabel("Fourier coefficients \u2192", fontsize=12)
legend(loc="upper right")
grid()

# Semilog plot of fourier coefficients of cos(cos(x)) using least Squares aproach
figure()
semilogy(range(51), abs(c_coscos), "go", label="Least Squares Approach")
semilogy(range(51), abs(coeff_coscos), "ro", label="True value")
xlabel("n \u2192", fontsize=12)
ylabel("Fourier coefficients \u2192", fontsize=12)
legend(loc="upper right")
grid()

# Loglog plot of fourier coefficients of cos(cos(x)) using least Squares aproach
figure()
loglog(range(51), abs(c_coscos), "go", label="Least Squares Approach")
loglog(range(51), abs(coeff_coscos), "ro", label="True value")
xlabel("n \u2192", fontsize=12)
ylabel("Fourier coefficients \u2192", fontsize=12)
legend(loc="upper right")
grid()

# Finding deviation and maximum deviation between the coe3fficients obtained by both the methods
coeffdiff_exp = abs(coeff_exp - c_exp)
coeffdiff_coscos = abs(coeff_coscos - c_coscos)

maxdev_exp = max(coeffdiff_exp)
maxdev_coscos = max(coeffdiff_coscos)

print("Largest deviation between coefficients for function e^(x) = ", maxdev_exp)
print("Largest deviation between coefficients for function coscos(x) = ", maxdev_coscos)
####


# Findiing the function values from least squares method
val_exp = A @ c_exp
val_coscos = A @ c_coscos
####

# Semilog plot of function e^(x) using Function Approximation
figure()
semilogy(x, e(period), "b-", label="Periodic Extension")
semilogy(x, e(x), "r", label="True Value")
semilogy(x1, val_exp, "go", label="Function Approximation")
xlabel("x \u2192", fontsize=12)
ylabel("e^(x) \u2192", fontsize=12)
legend(loc="upper right")
grid()

# Plot of function cos(cos(x)) using Function Approximation
figure()
plot(x, coscos(x), "r-", label="True value")
plot(x1, val_coscos, "go", label="Function Approximation")
xlabel("x \u2192", fontsize=12)
ylabel("cos(cos((x)) \u2192", fontsize=12)
legend(loc="upper right")
grid()

show()
