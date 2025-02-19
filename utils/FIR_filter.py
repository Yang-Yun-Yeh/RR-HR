import numpy as np
from sklearn.linear_model import LinearRegression

class FIR_filter:
    def __init__(self, _coefficients):
        self.ntaps = len(_coefficients)
        self.coefficients = _coefficients
        self.buffer = np.zeros(self.ntaps)

        # RLS
        self.P = None

        # LRLS
        self.n = None
        self.delta = None
        self.delta_D =None
        self.xi_b = None
        self.xi_f = None
        self.gamma = None
        self.b = None
        self.f = None

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
    
    # Recursive Least Squares
    def rls(self, alpha, delta, lam=0.9995):
        # Initialize
        if self.P is None:
            self.P = delta * np.eye(self.ntaps,self.ntaps)

        # Update
        g = self.P @ self.buffer / (lam + self.buffer @ self.P @ self.buffer) # (m, 1)
        g = g.reshape(-1, 1) # (m, 1)
        x_T = self.buffer.reshape(1, -1) # (1, m)
        self.P = self.P / lam - g @ x_T @ self.P / lam # (m, m)

        self.coefficients = self.coefficients + g.reshape(-1) * alpha

    # Solve sol. Implicitly
    def lrls(self, x, d, N, epsilon=1e-6, lam=0.9995):
        # Initialize (time n=0 is -1 in paper)(order m is same in paper)
        if self.delta is None:
            self.n = 0
            num = N + 1
            self.delta, self.delta_D = np.zeros((num, self.ntaps)), np.zeros((num, self.ntaps))
            self.xi_b, self.xi_f = np.zeros((num, self.ntaps+1)), np.zeros((num, self.ntaps+1))
            self.gamma = np.zeros((num, self.ntaps+1))
            self.b = np.zeros((num, self.ntaps+1))
            self.f = np.zeros((num, self.ntaps+1))

            self.xi_b[self.n, :], self.xi_f[self.n, :] = epsilon, epsilon
            self.gamma[self.n, :] = 1
            self.n += 1
        
        # Update
        self.gamma[self.n][0] = 1
        self.b[self.n][0], self.f[self.n][0] = x, x
        self.xi_b[self.n][0], self.xi_f[self.n][0] = x**2 + lam * self.xi_f[self.n-1][0], x**2 + lam * self.xi_f[self.n-1][0]
        e = d

        for m in range(self.ntaps):
            self.delta[self.n][m] = lam * self.delta[self.n-1][m] + self.b[self.n-1][m] * self.f[self.n][m] / self.gamma[self.n-1][m]
            self.gamma[self.n][m+1] = self.gamma[self.n][m] - self.b[self.n][m]**2 / self.xi_b[self.n][m]

            kappa_b = self.delta[self.n][m] / self.xi_f[self.n][m]
            kappa_f = self.delta[self.n][m] / self.xi_b[self.n-1][m]

            self.b[self.n][m+1] = self.b[self.n-1][m] - kappa_b * self.f[self.n][m]
            self.f[self.n][m+1] = self.f[self.n][m] - kappa_f * self.b[self.n-1][m]

            self.xi_b[self.n][m+1] = self.xi_b[self.n-1][m] - self.delta[self.n][m] * kappa_b
            self.xi_f[self.n][m+1] = self.xi_f[self.n][m] - self.delta[self.n][m] * kappa_f

            # Feedforward filtering
            self.delta_D[self.n][m] = lam * self.delta_D[self.n-1][m] + self.b[self.n][m] * e / self.gamma[self.n][m]
            kappa = self.delta_D[self.n][m] / self.xi_b[self.n][m]
            e = e - kappa * self.b[self.n][m]

        self.n += 1

        return e