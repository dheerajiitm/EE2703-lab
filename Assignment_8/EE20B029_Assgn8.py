from pylab import *

# function for writing printing the corresponding title of the function
def my_title(func):
    return f"Spectrum of {func}"


def sin5(x):
    return sin(5 * x)


def am(t):
    return (1 + 0.1 * cos(t)) * cos(10 * t)


def sin3(x):
    return (sin(x)) ** 3


def cos3(x):
    return (cos(x)) ** 3


def fm(x):
    return cos(20 * x + 5 * cos(x))


def gauss(x):
    return exp(-0.5 * x ** 2)


# Function which plots the spectrum of different functions
def dft(f, N, func, t_range, mod_lim, x_lim, w_range):
    t = linspace(-t_range, t_range, N + 1)[:-1]

    y = f(t)
    Y = fftshift(fft(y)) / N
    w = linspace(-w_range, w_range, N + 1)
    w = w[:-1]
    if f == gauss:
        Y = fftshift(abs(fft(y))) / N
        # Normalising for the case of gaussian
        Y = Y * sqrt(2 * pi) / max(Y)
        Y_ = exp(-(w ** 2) / 2) * sqrt(2 * pi)
        print(f"max error is {abs(Y - Y_).max()}")
    figure()
    subplot(2, 1, 1)
    plot(w, abs(Y), lw=2)
    xlim([-x_lim, x_lim])
    ylabel(r"$|y|$", size=16)
    title(my_title(func))
    grid(True)
    subplot(2, 1, 2)
    plot(w, angle(Y), "ro", lw=2)
    ii = where(abs(Y) > mod_lim)
    plot(w[ii], angle(Y[ii]), "go", lw=2)
    xlim([-x_lim, x_lim])
    ylabel(r"Phase of $Y$", size=16)
    xlabel(r"$omega$", size=16)
    grid(True)


# Plotting the functions already analysed in the assignment
dft(sin5, 128, "$sin(5t)$", 2 * pi, 1e-3, 15, 32)
dft(am, 512, "$(1 + 0.1cos(t))cos(10t)$", 4 * pi, 1e-3, 15, 64)


dft(cos3, 512, "$cos^3(t)$", 4 * pi, 1e-3, 15, 64)
dft(sin3, 512, "$sin^3(t)$", 4 * pi, 1e-3, 15, 64)
dft(fm, 512, "$cos(20t+5cos(t))$", 4 * pi, 1e-3, 40, 64)

# Plots with different time ranges
dft(gauss, 512, "$\exp(-t^2/2)$", 4 * pi, 1e-3, 10, 32)
dft(gauss, 512, "$\exp(-t^2/2)$", 8 * pi, 1e-3, 10, 32)
dft(gauss, 512, "$\exp(-t^2/2)$", 12 * pi, 1e-3, 10, 32)

# Plots with different sampling rates
dft(gauss, 256, "$\exp(-t^2/2)$", 8 * pi, 1e-3, 10, 32)
dft(gauss, 1024, "$\exp(-t^2/2)$", 8 * pi, 1e-3, 10, 32)


show()
