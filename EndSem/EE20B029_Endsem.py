#             APL EndSem
#             Analysis of antenna currents
#             Name: C. Dheeraj Sai
#             Roll No.: EE20B029


from pylab import *


# Defining all the parameters mentioned in the question paper.
l = 0.5
c = 2.9979e8
mu0 = 4e-7 * pi
N = 100
Im = 1
a = 0.01
wl = l * 4
f = c / wl
k = 2 * pi / wl
dz = l / N


##### Section 1 #####

# Defining the locations of all known and unknown currents.
z = linspace(-l, l, 2 * N + 1)

# Defining an array containing the locations of all unknown currents.
u = delete(z, [0, N, 2 * N])


##### Section 2 #####

# Defining a function which computes and returns a matrix to be used in solving the Ampere's Law
def Mat(N, a):
    M = (1 / (2 * pi * a)) * identity(2 * N - 2)
    return M


##### Section 3 #####

# Defining matrix Rz which contains the distances between all the known and unknown current locations
zi, zj = meshgrid(z, z)
Rz = sqrt((zi - zj) ** 2 + ones([2 * N + 1, 2 * N + 1], dtype=complex) * a ** 2)

# Defining matrix Ru whcih contains only the distances between unknown current locations.
ui, uj = meshgrid(u, u)
Ru = sqrt((ui - uj) ** 2 + ones([2 * N - 2, 2 * N - 2], dtype=complex) * a ** 2)

# Computing the matrices P and Pb using the vector potential to current relation.
j = 1j
RN = delete(Rz[N], [0, N, 2 * N])

P = (mu0 / (4 * pi)) * (exp(-k * Ru * j) / Ru) * dz
Pb = (mu0 / (4 * pi)) * (exp(-k * RN * j) / RN) * dz


##### Section 4 #####

# Computing the matrices Q and Qb  from the magntic field to current relation.
Q = -(a / mu0) * P * ((-k * j / Ru) + (-1 / Ru ** 2))
Qb = -(a / mu0) * Pb * ((-k * j / RN) + (-1 / RN ** 2))


##### Section 5 #####

# Obtaining the currents at unknown location by solving the matrix equation.
J = inv(Mat(N, a) - Q) @ Qb * Im

# Defining array containing current at all locations.
I = concatenate(([0], J[: N - 1], [Im], J[N - 1 :], [0]))

# Sinusoidal distribution assumption of current.
Ia = concatenate((Im * sin(k * (l - z[:N])), Im * sin(k * (l + z[N:]))))

figure(figsize=(8, 6))
plot(z, abs(I), lw=2, label="Calculated value ")
plot(z, abs(Ia), lw=2, label="Sinusoidal approximation")
xlabel(r"$z$")
ylabel(r"$I$")
title(f"Current distribution for N = {N}")
legend(loc="upper right")
grid()

show()

# print((z).round(2))
# print((u).round(2))
# print((Mat(N, a)).round(2))
# print((Rz).round(2))
# print((Ru).round(2))
# print((P * 1e8).round(2))
# print((Pb * 1e8).round(2))
# print((Q).round(2))
# print((Qb).round(2))
# print((J).round(2))
# print((I).round(2))
# print((Ia).round(2))
