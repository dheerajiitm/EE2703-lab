#             Assignment 7
#             A5: The Sympy module
#             Name: C. Dheeraj Sai
#             Roll No.: EE20B029


from pylab import *
from mpl_toolkits.mplot3d import Axes3D


##
## Example codes from the assignment
##

t = linspace(-pi, pi, 65)[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
y = sin(sqrt(2) * t)
y[0] = 0  # the sample corresponding to -tmax should be set zero
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y)) / 64.0
w = linspace(-pi * fmax, pi * fmax, 65)[:-1]

figure()
subplot(2, 1, 1)
plot(w, abs(Y), lw=2)
xlim([-10, 10])
ylabel(r"$|Y|$", size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
grid()
subplot(2, 1, 2)
plot(w, angle(Y), "ro", lw=2)
xlim([-10, 10])
ylabel(r"Phase of $Y$", size=16)
xlabel(r"$\omega$", size=16)
grid()


t1 = linspace(-pi, pi, 65)[:-1]
t2 = linspace(-3 * pi, -pi, 65)[:-1]
t3 = linspace(pi, 3 * pi, 65)[:-1]
# y=sin(sqrt(2)*t)
figure()
plot(t1, sin(sqrt(2) * t1), "b", lw=2)
plot(t2, sin(sqrt(2) * t2), "r", lw=2)
plot(t3, sin(sqrt(2) * t3), "r", lw=2)
ylabel(r"$y$", size=16)
xlabel(r"$t$", size=16)
title(r"$\sin\left(\sqrt{2}t\right)$")
grid()


t1 = linspace(-pi, pi, 65)[:-1]
t2 = linspace(-3 * pi, -pi, 65)[:-1]
t3 = linspace(pi, 3 * pi, 65)[:-1]
y = sin(sqrt(2) * t1)

figure()
plot(t1, y, "bo", lw=2)
plot(t2, y, "ro", lw=2)
plot(t3, y, "ro", lw=2)
ylabel(r"$y$", size=16)
xlabel(r"$t$", size=16)
title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
grid()

t = linspace(-pi, pi, 65)[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
y = t
y[0] = 0  # the sample corresponding to -tmax should be set zeroo
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y)) / 64.0
w = linspace(-pi * fmax, pi * fmax, 65)[:-1]

figure()
semilogx(abs(w), 20 * log10(abs(Y)), lw=2)
xlim([1, 10])
ylim([-20, 0])
xticks([1, 2, 5, 10], ["1", "2", "5", "10"], size=16)
ylabel(r"$|Y|$ (dB)", size=16)
title(r"Spectrum of a digital ramp")
xlabel(r"$\omega$", size=16)
grid()

t1 = linspace(-pi, pi, 65)[:-1]
t2 = linspace(-3 * pi, -pi, 65)[:-1]
t3 = linspace(pi, 3 * pi, 65)[:-1]
n = arange(64)
wnd = fftshift(0.54 + 0.46 * cos(2 * pi * n / 63))
y = sin(sqrt(2) * t1) * wnd
figure()
plot(t1, y, "bo", lw=2)
plot(t2, y, "ro", lw=2)
plot(t3, y, "ro", lw=2)
ylabel(r"$y$", size=16)
xlabel(r"$t$", size=16)
title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
grid()

t = linspace(-pi, pi, 65)[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
n = arange(64)
wnd = fftshift(0.54 + 0.46 * cos(2 * pi * n / 63))
y = sin(sqrt(2) * t) * wnd
y[0] = 0  # the sample corresponding to -tmax should be set zeroo
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y)) / 64.0
w = linspace(-pi * fmax, pi * fmax, 65)[:-1]

figure()
subplot(2, 1, 1)
plot(w, abs(Y), lw=2)
xlim([-8, 8])
ylabel(r"$|Y|$", size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
grid(True)
subplot(2, 1, 2)
plot(w, angle(Y), "ro", lw=2)
xlim([-8, 8])
ylabel(r"Phase of $Y$", size=16)
xlabel(r"$\omega$", size=16)
grid()

t = linspace(-4 * pi, 4 * pi, 257)[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
n = arange(256)
wnd = fftshift(0.54 + 0.46 * cos(2 * pi * n / 256))
y = sin(sqrt(2) * t)
y = y * wnd
y[0] = 0  # the sample corresponding to -tmax should be set zeroo
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y)) / 256.0
w = linspace(-pi * fmax, pi * fmax, 257)[:-1]

figure()
subplot(2, 1, 1)
plot(w, abs(Y), lw=2)
xlim([-8, 8])
ylabel(r"$|Y|$", size=16)
title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
grid()
subplot(2, 1, 2)
plot(w, angle(Y), "ro", lw=2)
xlim([-8, 8])
ylabel(r"Phase of $Y$", size=16)
xlabel(r"$\omega$", size=16)
grid()


# Function to plot the magmitude and phase plot.
def assymptotic_plots(w, Y, xlimit, Title, ylabel1, ylabel2, Xlabel):
    figure()
    subplot(2, 1, 1)
    plot(w, abs(Y), "b", lw=2)
    xlim([-xlimit, xlimit])
    ylabel(ylabel1, size=16)
    title(Title)
    grid()
    subplot(2, 1, 2)
    plot(w, angle(Y), "ro", lw=2)
    xlim([-xlimit, xlimit])
    ylabel(ylabel2, size=16)
    xlabel(Xlabel, size=16)
    grid()


y = cos(0.86 * t) ** 3
y1 = y * wnd
y[0] = 0
y1[0] = 0
y = fftshift(y)
y1 = fftshift(y1)
Y = fftshift(fft(y)) / 256.0
Y1 = fftshift(fft(y1)) / 256.0


# Plot a spectrum of cos^3(0.86t) without windowing
assymptotic_plots(
    w,
    Y,
    4,
    r"Spectrum of $\cos^{3}(0.86t)$ without Hamming window",
    r"$|Y|\rightarrow$",
    r"Phase of $Y\rightarrow$",
    r"$\omega\rightarrow$",
)

# Plot of spectrum of cos^3(0.86t) with windowing
assymptotic_plots(
    w,
    Y1,
    4,
    r"Spectrum of $\cos^{3}(0.86t)$ with Hamming window",
    r"$|Y|\rightarrow$",
    r"Phase of $Y\rightarrow$",
    r"$\omega\rightarrow$",
)


# Finding the values of w0 and delta from the spectrum of the signal
w0 = 1.5
d = 0.5

t = linspace(-pi, pi, 129)[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
wnd = fftshift(0.54 + 0.46 * cos(2 * pi * arange(128) / 128))
y = cos(w0 * t + d) * wnd
y[0] = 0
y = fftshift(y)
Y = fftshift(fft(y)) / 128.0
w = linspace(-pi * fmax, pi * fmax, 129)
w = w[:-1]
assymptotic_plots(
    w,
    Y,
    4,
    r"Spectrum of $\cos(w_0t+\delta)$ with Hamming window",
    r"$|Y|\rightarrow$",
    r"Phase of $Y\rightarrow$",
    r"$\omega\rightarrow$",
)


# w0 is calculated by finding the weighted average of all w>0.
ii = where(w > 0)
omega = sum(abs(Y[ii]) ** 2 * w[ii]) / sum(abs(Y[ii]) ** 2)  # weighted average
print(f"Estimated value of w0 without noise = {omega}")
# Delta is found by calculating the phase at w closest to w0.
i = abs(w - omega).argmin()
delta = angle(Y[i])
print(f"Estimated value of delta without noise = {delta}")


# Finding the values of w0 and delta from the spectrum of a noisy signal
y = (cos(w0 * t + d) + 0.1 * randn(128)) * wnd
y[0] = 0
y = fftshift(y)
Y = fftshift(fft(y)) / 128.0
assymptotic_plots(
    w,
    Y,
    4,
    r"Spectrum of a noisy $\cos(w_0t+\delta)$ with Hamming window",
    r"$|Y|\rightarrow$",
    r"Phase of $Y\rightarrow$",
    r"$\omega\rightarrow$",
)

ii = where(w >= 0)
w_avg = sum(abs(Y[ii]) ** 2 * w[ii]) / sum(abs(Y[ii]) ** 2)
i = abs(w - w_avg).argmin()
delta = angle(Y[i])
print("Estimated value of w0 with noise: ", w_avg)
print("Estimated value of delta with noise: ", delta)


# Plotting spectrum of chirped signal
t = linspace(-pi, pi, 1025)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1 / dt
n = arange(1024)
wnd = fftshift(0.54 + 0.46 * cos(2 * pi * n / 1024))
y = cos(16 * t * (1.5 + t / (2 * pi))) * wnd
y[0] = 0
y = fftshift(y)
Y = fftshift(fft(y)) / 1024.0
w = linspace(-pi * fmax, pi * fmax, 1025)
w = w[:-1]
assymptotic_plots(
    w,
    Y,
    100,
    r"Spectrum of chirped function",
    r"$|Y|\rightarrow$",
    r"Phase of $Y\rightarrow$",
    r"$\omega\rightarrow$",
)


# Surface plot with respect to t and w
t_arr = split(t, 16)
Y_mag = zeros((16, 64))
Y_phase = zeros((16, 64))

for i in range(len(t_arr)):
    n = arange(64)
    wnd = fftshift(0.54 + 0.46 * cos(2 * pi * n / 64))
    y = cos(16 * t_arr[i] * (1.5 + t_arr[i] / (2 * pi))) * wnd
    y[0] = 0
    y = fftshift(y)
    Y = fftshift(fft(y)) / 64.0
    Y_mag[i] = abs(Y)
    Y_phase[i] = angle(Y)

t = t[::64]
w = linspace(-fmax * pi, fmax * pi, 64 + 1)
w = w[:-1]
t, w = meshgrid(t, w)

fig = figure(13)
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(w, t, Y_mag.T, cmap=cm.jet, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title("surface plot")
ylabel(r"$\omega\rightarrow$")
xlabel(r"$t\rightarrow$")

show()
