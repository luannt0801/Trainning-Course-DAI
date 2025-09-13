import numpy as np, matplotlib.pylab as plt, time

class lab2partA1():
    def __init__(self):
        self.sleep = 0.2
        self.low = -20
        self.up = 10
        
    def initialize(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(r"Parameter $a$")
        self.ax.set_ylabel(r"$F(a)$")
        
    def plot(self, i, func, a):
        if i == 0: self.initialize()
        
        x = np.arange(self.low, self.up, 0.1)
        y = [func(v) for v in x]
        self.ax.plot(x, y)
        
        fa = func(a)
        self.ax.scatter([a], [fa], marker="o")
        self.ax.set_title("$a$ = {0:.9f}, $F(a)$ = {1:.9f} ".format(a, fa))
        
        self.fig.canvas.draw()
        time.sleep(self.sleep)

class lab2partA2():
    def __init__(self):
        self.sleep = 0.2
        self.low = -100
        self.up = 100
            
    def initialize(self):
        self.fig, self.ax = plt.subplots()
        
        self.ax.set_xlabel(r"Parameter $a$")
        self.ax.set_ylabel(r"Parameter $b$")
        self.ax.set_title(r"$F(a, b)$")
        
        x = np.arange(self.low, self.up, 0.1)
        y = np.arange(self.low, self.up, 0.1)

        X, Y = np.meshgrid(x, y)
        Z = 5 + np.square(X) + 1.5 * np.square(Y) + (X*Y)
        
        self.ax.contour(X, Y, Z)
        
    def plot(self, i, func, a, b):
        if i == 0: self.initialize()
        
        self.ax.scatter([a], [b], marker="o")
        fab = func(a, b)
        self.ax.set_title(r"$a$ = {0:.9f}, $b$ = {1:.9f}, $F(a, b)$ = {2:.9f} ".format(a, b, fab))
        
        self.fig.canvas.draw()
        time.sleep(self.sleep)

class lab2partB1():
    def __init__(self):
        self.sleep = 0.2
        self.low = -100
        self.up = 100        
        self.mses = []
        
    def initialization(self, X, y):
        self.mses = []
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        self.ax1.set_xlabel(r"Feature $x_1$")
        self.ax1.set_ylabel(r"$h_{\theta}(x)$")
        
        self.ax2.set_xlabel("Number of iterations")
        self.ax2.set_ylabel(r"MSE (i.e. $E(\theta)$)")
        
        # min_y, max_y = min(y), max(y)
        min_x, max_x = np.min(X), np.max(X)
        self.low = min_x # min(min_y, min_x)
        self.up = max_x # max(max_y, max_x)
        
    def plot(self, i, func, theta, X, y):
        if i == 0: self.initialization(X, y)
            
        mse = func(theta, X, y)
        
        self.mses.append(mse)
        if len(self.mses)%100 != 0: return
        
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.set_xlabel("Population of City in 10,000s")
        self.ax1.set_ylabel("Profit in $10,000s$")
        self.ax2.set_xlabel("Number of iterations")
        self.ax2.set_ylabel(r"Cost $E(\theta)$")

        
        self.ax1.scatter([x[1] for x in X], y, marker="+", color="red", label="Training Data")
        self.ax2.plot(self.mses if len(self.mses)<5 else self.mses[5:], color="green")
        
        xxxline = np.linspace(self.low, self.up, 100)
        yyyline = theta[0] + theta[1] * xxxline
        self.ax1.plot(xxxline, yyyline, label="Linear Regression")
        
        self.ax1.set_title(r"$\theta_0$ = {0:.5f}, $\theta_1$ = {1:.5f}".format(theta[0], theta[1]))
        self.ax2.set_title(r"$E(\theta)$ = {0:.5f}".format(mse))
        
        #plt.legend()
        self.fig.canvas.draw()
        time.sleep(self.sleep)
