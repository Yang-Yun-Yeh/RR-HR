import numpy as np
from sklearn.linear_model import LinearRegression

class FIR_filter:
    def __init__(self, _coefficients):
        self.ntaps = len(_coefficients)
        self.coefficients = _coefficients
        self.buffer = np.zeros(self.ntaps)

        # RLS
        self.P = None

    def filter(self, v):
        for j in range(self.ntaps-1):
            self.buffer[self.ntaps-j-1] = self.buffer[self.ntaps-j-2]
        self.buffer[0] = v
        return np.inner(self.buffer, self.coefficients)
    
    # Least Mean Suqare
    def lms(self, error, mu=0.01):
        for j in range(self.ntaps):
            self.coefficients[j] = self.coefficients[j] + error * mu * self.buffer[j]
    
    # Least Square
    def ls(self, x, d):
        n, m = len(x), self.ntaps
        A = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if i - j >= 0:
                    A[i][j] = x[i-j]
        
        h_hat = np.linalg.inv(A.T @ A) @ A.T @ d
        # print(f'h_hat:{h_hat}')

        # LR with sklearn
        reg = LinearRegression(fit_intercept=False).fit(A, d)
        # print(f'coef:{reg.coef_}')

        # self.coefficients = h_hat
        self.coefficients = reg.coef_
    
    def rls(self, alpha, delta, lam=0.9995):
        if self.P is None:
            self.P = delta * np.eye(self.ntaps,self.ntaps)

        g = self.P @ self.buffer / (lam + self.buffer @ self.P @ self.buffer) # (m, 1)
        g = g.reshape(-1, 1) # (m, 1)
        x_T = self.buffer.reshape(1, -1) # (1, m)
        self.P = self.P / lam - g @ x_T @ self.P / lam # (m, m)

        self.coefficients = self.coefficients + g.reshape(-1) * alpha