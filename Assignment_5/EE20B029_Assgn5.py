#             Assignment 5
#             A5: The Resistor Problems
#             Name: C. Dheeraj Sai
#             Roll No.: EE20B029

from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import sys
from scipy.linalg import lstsq


def update_phi(phi, phiold):
    phi[1:-1, 1:-1] = 0.25 * (
        phiold[1:-1, 0:-2] + phiold[1:-1, 2:] + phiold[0:-2, 1:-1] + phiold[2:, 1:-1]
    )
    return phi


def update_boundary(phi, ind):
    phi[:, 0] = phi[:, 1]  #             Left Boundary
    phi[:, Nx - 1] = phi[:, Nx - 2]  #   Right Boundary
    phi[0, :] = phi[1, :]  #             Top Boundary
    phi[Ny - 1, :] = 0  #                Bottom Boundary
    phi[ind] = 1.0
    return phi


def error_fit(x, y):
    lny = log(y)
    x_vec = zeros((len(x), 2))
    x_vec[:, 0] = 1
    x_vec[:, 1] = x
    return lstsq(x_vec, lny)[0]


def max_error(lnA, B, Niter):
    return -exp(lnA + B * (Niter + 0.5)) / B


# Usage
print(f"Usage: python {sys.argv[0]} <Nx> <Ny> <radius> <Niter>")

# Taking input
if len(sys.argv) == 5:
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
    radius = int(sys.argv[3])
    Niter = int(sys.argv[4])
else:
    Nx = 25
    Ny = 25
    radius = 8
    Niter = 1500


phi = zeros((Nx, Ny), dtype=float)
x = linspace(-0.5, 0.5, num=Nx, dtype=float)
y = linspace(-0.5, 0.5, num=Ny, dtype=float)
Y, X = meshgrid(y, x, sparse=False)
ind = where(X ** 2 + Y ** 2 < (0.35) ** 2)
phi[ind] = 1.0

# contour plot of potential
figure()
contourf(X, Y, phi)
colorbar()
plot((ind[0] + 0.5 - Nx / 2) / Nx, (ind[1] + 0.5 - Ny / 2) / Ny, "ro", label="V = 1")
title("Potential Configuration", fontsize=16)
xlabel("x \u2192", fontsize=12)
ylabel("y \u2192", fontsize=12)
legend()

# Calculating error
err = np.zeros(Niter)
for k in range(Niter):
    phiold = phi.copy()
    phi = update_phi(phi, phiold)
    phi = update_boundary(phi, ind)
    err[k] = (abs(phi - phiold)).max()


x = asarray(range(Niter))

# semilog plot of error
figure()
semilogy(x, err, label="real")
semilogy(x[::50], err[::50], "ro", label="every 50th value")
title("Error on a semilog plot", fontsize=16)
xlabel("No of iterations", fontsize=12)
ylabel("Error", fontsize=12)

# loglog plot of error
figure()
loglog(x, err, label="real")
loglog(x[::50], err[::50], "ro", label="every 50th value")
title("Error on a loglog plot", fontsize=16)
xlabel("No of iterations", fontsize=12)
ylabel("Error", fontsize=12)
legend(loc="upper right")


# Getting the best fit for the error values
lnA, B = error_fit(range(Niter), err)  # fit1
lnA_500, B_500 = error_fit(range(Niter)[500:], err[500:])  # fit2

# loglog plot of error, fit1, fit2
figure()
loglog(x, err, label="errors")
loglog(x[::50], exp(lnA + B * asarray(range(Niter))[::50]), "ro", label="fit1")
loglog(
    x[500::50],
    exp(lnA_500 + B_500 * asarray(range(Niter))[500::50]),
    "go",
    label="fit2",
)
title("Best fit for error on a loglog scale", fontsize=16)
xlabel("No of iterations", fontsize=12)
ylabel("Error", fontsize=12)
legend(loc="upper right")

# semilog plot of error, fit1, fit2
figure()
semilogy(x, err, label="errors")
semilogy(x[::50], exp(lnA + B * asarray(range(Niter))[::50]), "ro", label="fit1")
semilogy(
    x[500::50],
    exp(lnA_500 + B_500 * asarray(range(Niter))[500::50]),
    "go",
    label="fit2",
)
title("Best fit for error on a semilog scale", fontsize=16)
xlabel("No of iterations", fontsize=12)
ylabel("Error", fontsize=12)
legend(loc="upper right")


# semilog plot of max error
figure()
semilogy(x[500::50], max_error(lnA_500, B_500, x[500::50]), "ro")
title("Semilog plot of Cumulative Error vs number of iterations", fontsize=16)
xlabel("No of iterations", fontsize=12)
ylabel("Error", fontsize=12)


# 3-D plot of potential
fig1 = plt.figure(0)
ax = p3.Axes3D(fig1, auto_add_to_figure=False)
fig1.add_axes(ax)
title("The 3-D  surface plot of the potential", fontsize=16)
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=cm.jet)


# Contour plot of potential
figure()
contourf(Y, X[::-1], phi)
colorbar()
title("2D Contour plot of potential", fontsize=16)
xlabel("X", fontsize=12)
ylabel("Y", fontsize=12)

# Current vectors
Jx = 1 / 2 * (phi[1:-1, 2:] - phi[1:-1, :-2])
Jy = 1 / 2 * (phi[2:, 1:-1] - phi[:-2, 1:-1])


# Vector plot of current flow
figure()
quiver(Y[1:-1, 1:-1], -X[1:-1, 1:-1], Jx[:, ::-1], Jy)
plot((ind[0] + 0.5 - Nx / 2) / Nx, (ind[1] + 0.5 - Ny / 2) / Ny, "ro")
title("The vector plot of the current flow", fontsize=16)
xlabel("X", fontsize=12)
ylabel("Y", fontsize=12)


show()
