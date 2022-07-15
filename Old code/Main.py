# Assignment 7
# Importing Libraries
import myLib
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


dxl = [0.5, 0.25, 0.05, 0.02]    #dx different values

# The differential functions
def dy_dx_1(x, y):
    return y * math.log(y)/x

def dy_dx_2(x, y):
    return 6 - (2*y/x)


# Q1 (Euler)
# Equation 1
for dx in dxl:
    xl, yl = myLib.forwardEuler(dy_dx_1, (0, 10), (2, math.e), dx)
    plt.plot(xl, yl, ".", label = f"dx = {dx}")

# Analytical solution (by hand)
def AnalytQ1a(x):
    return math.e**(x/2)

dx = dxl[-1]
xlAnalyt = [0]
ylAnalyt = [AnalytQ1a(xl[0])]
while xlAnalyt[-1] < 10:
    xlAnalyt.append(xlAnalyt[-1] + dx)
    ylAnalyt.append(AnalytQ1a(xlAnalyt[-1]))

# Plotting the curves
plt.plot(xlAnalyt, ylAnalyt, label = "y = exp(x/2)")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Q1a")
plt.savefig("Q1a.pdf")		#Saving the plot
plt.show()
plt.clf()

# Equation 2
for dx in dxl:
    xl, yl = myLib.forwardEuler(dy_dx_2, (1, 10), (3, 1), dx)
    plt.plot(xl, yl, ".", label = f"dx = {dx}")

# Analytical solution (by hand)
def AnalytQ1b(x):
    return 2*x - (45/(x**2))

dx = dxl[-1]
xlAnalyt = [1]
ylAnalyt = [AnalytQ1b(xl[0])]
while xlAnalyt[-1] < 10:
    xlAnalyt.append(xlAnalyt[-1] + dx)
    ylAnalyt.append(AnalytQ1b(xlAnalyt[-1]))

# Plotting the curves
plt.plot(xlAnalyt, ylAnalyt, label = "y = 2x - 45/x^2")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Q1b")
plt.savefig("Q1b.pdf")		#Saving the plot
plt.show()
plt.clf()


# Q2
def dz_dx(x, z):        # Where, z = dy/dx
    return 1 - x - z

# Runge Kutta
for dx in dxl:
    xl, zl = myLib.RK4(dz_dx, (-5 - dx, 5 + dx), (0, 1), dx=dx / 2)
    xl = list(map(lambda x: round(x, 6), xl))

    def dy_dx(x, y):
        return zl[xl.index(round(x, 6))]

    xl, yl = myLib.RK4(dy_dx, (-5, 5), (0, 2), dx=dx)
    plt.plot(xl, yl, ".", label = f"dx = {dx}")

# Analytical solution (by hand)
def AnalytQ2(x):
    return 1 + math.e**(-x) - (x**2)/2 + 2*x

xl = [-5]
yl = [AnalytQ2(xl[0])]
while xl[-1] < 5:
    xl.append(xl[-1] + dx)
    yl.append(AnalytQ2(xl[-1]))

# Plotting the curves
plt.plot(xl, yl, label = "analytical sol.")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Q2")
plt.savefig("Q2.pdf")		#Saving the plot
plt.show()
plt.clf()


# Q3 (Shooting Method)
def dz_dx(x, z):
    return z + 1

root1 = [0, 1]
root2 = [1, 2 * (math.e - 1)]
for dx in dxl[::-1]:
    xl, yl = myLib.shootingMethod(dz_dx, root1, root2, odeSolver="RK4", iterationLimit = 10, dx = dx)
    plt.plot(xl, yl, ".", label = f"dx = {dx}")

def y(x):
    return 2*math.e**x - 1 - x

yl = [1]
xl = [0]
dx = 0.01
for i in range(int(1/dx)):
    xl.append(xl[-1] + dx)
    yl.append(y(xl[-1]))

# Plotting the curves
plt.plot(xl, yl, label = "solution")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Q3")
plt.savefig("Q3.pdf")		#Saving the plot
plt.show()
plt.clf()
