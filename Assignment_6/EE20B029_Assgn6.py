#             Assignment 5
#             A5: The Resistor Problems
#             Name: C. Dheeraj Sai
#             Roll No.: EE20B029


from pylab import *
import scipy.signal as sp


# Function gives numerator and denominator of the decay function in laplace domain
def laplace_fraction(decay, omega):
    num = np.poly1d([1, omega])
    den = np.poly1d([1, (2 * omega), ((omega * omega) + (decay * decay))])
    return num, den


# Finding the output x(t)
num1, den1 = laplace_fraction(0.5, 1.5)
spring_transform = poly1d([1, 0, 2.25])  # transfer function of the spring
t = linspace(0, 50, 500)
H1 = sp.lti(num1, polymul(den1, spring_transform))
t1, x1 = sp.impulse(H1, None, t)

# Output plot at alpha = 0.5
figure()
plot(t1, x1)
title("alpha = 0.5")
xlabel("time", fontsize=12)
ylabel("x", fontsize=12)
grid()

# Output x(t) with a slower decay
num2, den2 = laplace_fraction(0.05, 1.5)
H2 = sp.lti(num2, polymul(den2, spring_transform))
t2, x2 = sp.impulse(H2, None, t)

# Output plot at alpha = 0.05
figure()
plot(t2, x2)
title("alpha = 0.05")
xlabel("time", fontsize=12)
ylabel("x", fontsize=12)
grid()

# Output plots for different frequencies
H = sp.lti([1], spring_transform)
for omega in arange(1.4, 1.6, 0.05):
    t0 = linspace(0, 100, 500)
    func = cos(omega * t0) * exp(-0.05 * t0)
    t0, x, svec = sp.lsim(H, func, t0)
    figure()
    plot(t0, x)
    title(f"$\omega = {omega}$")
    xlabel("time", fontsize=12)
    ylabel("x", fontsize=12)
    grid()

# Solving for coupled spring distortion functions using laplace transforms
t = linspace(0, 20, 500)
x_laplace = sp.lti(np.poly1d([1, 0, 2]), np.poly1d([1, 0, 3, 0]))
y_laplace = sp.lti(np.poly1d([2]), np.poly1d([1, 0, 3, 0]))
tx, x = sp.impulse(x_laplace, None, t)
ty, y = sp.impulse(y_laplace, None, t)
figure()
title("Coupled spring")
plot(tx, x, label="x(t)")
plot(ty, y, label="y(t)")
xlabel("time", fontsize=12)
ylabel("x(t), y(t)", fontsize=12)
legend(loc="upper right")

# Magnitude bode plot of transfer function of a two port network
H = sp.lti([1], [1e-12, 1e-4, 1])
w, S, phi = H.bode()
figure()
semilogx(w, S)
title("Magnitude bode plot")
xlabel("$\omega$", fontsize=12)
ylabel("$\log|H(s)|$", fontsize=12)
grid()

# Phase bode plot of transfer function of a two port network
figure()
semilogx(w, phi)
title("Phase bode plot")
xlabel("$\omega$", fontsize=12)
ylabel(r"$\angle H(s)$", fontsize=12)
grid()

# Calculating output voltage using input voltage and transfer function
t = linspace(0, 30e-6, 500)
vi = cos(1e3 * t) - cos(1e6 * t)
t, vo, svec = sp.lsim(H, vi, t)

# Plot of Output Voltage for 0 < t < 30µs
figure()
plot(t, vo)
title("The Output Voltage for 0 < t < 30µs")
xlabel("t", fontsize=12)
ylabel("Vo(t)", fontsize=12)
grid()

t = linspace(0, 10e-3, 10000)
vi = cos(1e3 * t) - cos(1e6 * t)
t, vo, svec = sp.lsim(H, vi, t)

# Plot of Output Voltage till t = 10msec
figure()
plot(t, vo)
title("The Output Voltage till t = 10msec")
xlabel("t", fontsize=12)
ylabel("Vo(t)", fontsize=12)
grid()


show()
