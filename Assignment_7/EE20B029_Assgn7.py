#             Assignment 7
#             A5: The Sympy module
#             Name: C. Dheeraj Sai
#             Roll No.: EE20B029


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
from sympy import *


init_session
s = symbols("s")

# Function for lowpass filter
def lowpass(R1, R2, C1, C2, G, Vi):
    s = symbols("s")
    A = Matrix(
        [
            [0, 0, 1, -1 / G],
            [-1 / (1 + s * R2 * C2), 1, 0, 0],
            [0, -G, G, 1],
            [-(1 / R1) - (1 / R2) - (s * C1), 1 / R2, 0, s * C1],
        ]
    )
    b = Matrix([0, 0, 0, -Vi / R1])
    V = A.inv() * b
    return (A, b, V)


# Function for highpass filter
def highpass(R1, R3, C1, C2, G, Vi):
    s = symbols("s")
    A = Matrix(
        [
            [0, 0, 1, -1 / G],
            [-(s * R3 * C2) / (1 + s * R3 * C2), 1, 0, 0],
            [0, -G, G, 1],
            [-(s * C1) - (s * C2) - (1 / R1), s * C2, 0, 1 / R1],
        ]
    )
    b = Matrix([0, 0, 0, -Vi * s * C1])
    V = A.inv() * b
    return (A, b, V)


# For getting coefficients of polynomials
def poly_to_list(num, den):
    isFloat = False
    try:
        num = Poly(num).all_coeffs()
    except GeneratorsNeeded:
        num = num
        isFloat = True
    den = Poly(den).all_coeffs()
    den_list = []
    num_list = []
    for i in den:
        den_list.append(float(i))
    den_list = np.array(den_list)

    if isFloat:
        num_list = num
    else:
        for i in num:
            num_list.append(float(i))
        num_list = np.array(num_list)
    return num_list, den_list


# Function for time domain output.
def output(Vo, t, inc):
    num = fraction(simplify(Vo))[0]
    den = fraction(simplify(Vo))[1]
    num_list, den_list = poly_to_list(num, den)

    num_list = np.poly1d(num_list)
    den_list = np.poly1d(den_list)

    Y = sp.lti(num_list, den_list)
    t = np.linspace(0.0, t, inc)
    t, y = sp.impulse(Y, None, t)
    return t, y


# Function for simulating the output in freuency domain
def output_simul(H, x, t):
    num = fraction(simplify(H))[0]
    den = fraction(simplify(H))[1]
    num_list, den_list = poly_to_list(num, den)

    num2 = np.poly1d(num_list)
    den2 = np.poly1d(den_list)

    H = sp.lti(num2, den2)
    t, y, sec = sp.lsim(H, x, t)
    return t, y


Vi = 1
A, b, V = lowpass(10000, 10000, 1e-9, 1e-9, 1.586, Vi)
Vo1 = V[3]
w = np.logspace(0, 8, 801)
ss = 1j * w
hf = lambdify(s, Vo1, "numpy")
v = abs(hf(ss))
ph = np.angle(v)

plt.figure()
plt.title("Magnitude bode plot")
plt.loglog(w, v)
plt.grid()
plt.xlabel("Frequency in log scale")
plt.ylabel("Magnitude in log scale")


plt.figure()
plt.title("Phase bode plot")
plt.semilogx(w, ph)
plt.grid()
plt.xlabel("Frequency in log scale")
plt.ylabel("Phase in radians")

Vi = 1 / s
A, b, V = lowpass(10000, 10000, 1e-9, 1e-9, 1.586, Vi)
Vo2 = V[3]
w = np.logspace(0, 8, 801)
ss = 1j * w
hf = lambdify(s, Vo2, "numpy")
v = abs(hf(ss))
ph = np.angle(v)
t, y = output(Vo2, 4e-3, 10001)

plt.figure()
plt.loglog(w, v)
plt.grid()
plt.title("Step Response of the low pass filter")
plt.xlabel("Frequency in log scale")
plt.ylabel("Magnitude in log scale")

plt.figure()
plt.plot(t, y)
plt.grid()
plt.title("Time domain output to step input")
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.ylim(-1, 1)


t = np.linspace(0.0, 4e-3, 100001)
x = np.sin(2000 * np.pi * t) + np.cos(2 * (10 ** 6) * np.math.pi * t)


plt.figure()
plt.plot(t, x)
plt.grid()
plt.title("Input to lowpass filter")
plt.xlabel("Time")
plt.ylabel("y(t)")

t, y = output_simul(Vo1, x, t)

plt.figure()
plt.plot(t, y)
plt.grid()
plt.title("Output of lowpass filter to given input")
plt.xlabel("Time")
plt.ylabel("y(t)")

Vi = 1
A, b, V = highpass(10000, 10000, 1e-9, 1e-9, 1.586, Vi)
Vo1 = V[3]
w = np.logspace(0, 8, 801)
ss = 1j * w
hf = lambdify(s, Vo1, "numpy")
v = abs(hf(ss))
ph = np.angle(v)

plt.figure()
plt.title("Magnitude bode plot")
plt.loglog(w, v)
plt.grid()
plt.xlabel("Frequency in log scale")
plt.ylabel("Magnitude in log scale")


plt.figure()
plt.title("Phase bode plot")
plt.semilogx(w, ph)
plt.grid()
plt.xlabel("Frequency in log scale")
plt.ylabel("Phase in radians")

Vi = 1 / s
A, b, V = highpass(10000, 10000, 1e-9, 1e-9, 1.586, Vi)
Vo2 = V[3]
w = np.logspace(0, 8, 801)
ss = 1j * w
hf = lambdify(s, Vo2, "numpy")
v = abs(hf(ss))
ph = np.angle(v)
t, y = output(Vo2, 4e-3, 10001)

plt.figure()
plt.loglog(w, v)
plt.grid()
plt.title("Step Response of the high pass filter")
plt.xlabel("Frequency in log scale")
plt.ylabel("Magnitude in log scale")

plt.figure()
plt.plot(t, y)
plt.grid()
plt.title("Time domain output to step input")
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.ylim(-1, 1)


t = np.linspace(0.0, 4e-3, 100001)
x = np.sin(2000 * np.pi * t) + np.cos(2 * (10 ** 6) * np.math.pi * t)


plt.figure()
plt.plot(t, x)
plt.grid()
plt.title("Input to highpass filter")
plt.xlabel("Time")
plt.ylabel("y(t)")

t, y = output_simul(Vo1, x, t)

plt.figure()
plt.plot(t, y)
plt.grid()
plt.title("Output of highpass filter to given input")
plt.xlabel("Time")
plt.ylabel("y(t)")

t = np.linspace(0.0, 4e-3, 100001)
x = (np.sin(2000 * np.pi * t)) * np.exp((-(10 ** 3)) * t)

plt.figure()
plt.plot(t, x)
plt.grid()
plt.title("Damped sinusoidal input to highpass filter")
plt.xlabel("Time")
plt.ylabel("y(t)")

t, y = output_simul(Vo1, x, t)

plt.figure()
plt.plot(t, y)
plt.grid()
plt.title("Output of highpass filter to given input")
plt.xlabel("Time")
plt.ylabel("y(t)")

t = np.linspace(0.0, 4e-3, 100001)
x = (np.sin(2000000 * np.pi * t)) * np.exp((-(10 ** 4)) * t)

plt.figure()
plt.plot(t, x)
plt.grid()
plt.title("Damped sinusoidal input to highpass filter")
plt.xlabel("Time")
plt.ylabel("y(t)")

t, y = output_simul(Vo1, x, t)

plt.figure()
plt.plot(t, y)
plt.grid()
plt.title("Output of highpass filter to given input")
plt.xlabel("Time")
plt.ylabel("y(t)")

plt.show()
