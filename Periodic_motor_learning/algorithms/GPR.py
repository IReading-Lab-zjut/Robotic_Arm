# This is the original GP file
import autograd.numpy as np
from autograd import value_and_grad, grad
from scipy.optimize import minimize
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve
np.random.seed(5)


class sgpr:
    def __init__(self, X, y, likelihood_noise=0.1, restart=1):
        self.X = X
        self.y = y
        self.init_param = []
        self.param = []
        self.input_dim = np.shape(self.X)[1]
        self.input_num = np.shape(self.X)[0]
        self.likelihood_noise = likelihood_noise
        self.restart = restart
        self.cov_y_y = None
        self.beta = None

    def init_random_param(self):
        kern_length_scale = 0.01 * np.random.normal(size=self.input_dim) + 2
        kern_noise = 0.1 * np.random.normal(size=1)
        self.init_param = np.hstack((kern_noise, kern_length_scale))
        self.param = self.init_param.copy()
        # print("self.init_param is", self.init_param)

    def set_param(self, param):
        self.param = param.copy()
        self.cov_y_y = self.rbf(self.X, self.X, self.param) + self.likelihood_noise ** 2 * np.eye(self.input_num)
        self.beta = solve(self.cov_y_y, self.y)
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

    def save_param(self, direction):
        np.savetxt(direction, self.param)

    def set_XY(self, X, y):
        self.X = X
        self.y = y
        self.input_dim = np.shape(self.X)[1]
        self.input_num = np.shape(self.X)[0]

    def build_objective(self, param):
        cov_y_y = self.rbf(self.X, self.X, param)
        cov_y_y = cov_y_y + self.likelihood_noise**2 * np.eye(self.input_num)
        out = - mvn.logpdf(self.y, np.zeros(self.input_num), cov_y_y)
        return out

    def train(self):
        max_logpdf = -1e20
        # cons = con((0.001, 10))
        for i in range(self.restart):
            self.init_random_param()
            result = minimize(value_and_grad(self.build_objective), self.init_param, jac=True, method='L-BFGS-B', tol=0.01)
            logpdf = -result.fun
            param = result.x
            if logpdf > max_logpdf:
                self.param = param
                max_logpdf = logpdf
        # pre-computation
        self.cov_y_y = self.rbf(self.X, self.X, self.param) + self.likelihood_noise**2 * np.eye(self.input_num)
        self.beta = solve(self.cov_y_y, self.y)
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

    def rbf(self, x, x_, param):
        kern_noise = param[0]
        sqrt_kern_length_scale = param[1:]
        diffs = np.expand_dims(x / sqrt_kern_length_scale, 1) - np.expand_dims(x_ / sqrt_kern_length_scale, 0)
        return kern_noise**2 * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def predict_determined_input(self, inputs):  # 单维GP预测
        cov_y_f = self.rbf(self.X, inputs, self.param)
        mean_outputs = np.dot(cov_y_f.T, self.beta.reshape((-1, 1)))
        var = (self.param[0]**2 - np.diag(np.dot(np.dot(cov_y_f.T, self.inv_cov_y_y), cov_y_f))).reshape(-1, 1)
        return mean_outputs, var

    def print_params(self):
        print('final param is', self.param)

    def callback(self, param):
        # ToDo: add something you want to know about the training process
        pass

    def gradient2input(self, input):
        sqrt_kern_length_scale = self.param[1:]
        temp1 = np.dot(self.X - input, np.diag(1 / (sqrt_kern_length_scale**2)))
        cov_y_f = self.rbf(self.X, input.reshape(1, -1), self.param).reshape(-1)
        temp2 = (temp1.T * cov_y_f).T
        gradient = np.dot(self.beta.T, temp2)
        return gradient.reshape(-1)


# multi GP
class mgpr():
    def __init__(self, X, Y, likelihood_noise=0.1, restart=1):
        self.X = X
        self.Y = Y
        self.param = []
        self.input_dim = np.shape(X)[1]
        self.input_num = np.shape(X)[0]
        self.output_dim = np.shape(Y)[1]
        self.likelihood_noise = np.zeros(self.output_dim) + likelihood_noise
        self.restart = restart

    def set_XY(self, X, Y):
        self.X = X
        self.Y = Y
        self.input_num = np.shape(X)[0]
        for i in range(self.output_dim):
            self.models[i].set_XY(self.X, self.Y[i])

    def set_param(self, param):
        self.create_models()
        self.param = param.copy()
        for i in range(self.output_dim):
            self.models[i].set_param(self.param[i])

    def save_param(self, direction):
        np.savetxt(direction, self.param)

    def create_models(self):
        self.models = []
        for i in range(self.output_dim):
            self.models.append(sgpr(self.X, self.Y[:, i], likelihood_noise=self.likelihood_noise[i], restart=self.restart))

    def init_random_param(self):
        for i in range(self.output_dim):
            self.models[i].init_random_param()

    def train(self):
        self.create_models()
        self.init_random_param()
        for i in range(self.output_dim):
            self.models[i].train()
            if i == 0:
                self.param = self.models[i].param.copy()
            else:
                self.param = np.vstack((self.param, self.models[i].param.copy()))

    def print_params(self):
        print('final param is', self.param)

    def predict_determined_input(self, inputs):
        mean_outputs0, var0 = self.models[0].predict_determined_input(inputs)
        mean_outputs1, var1 = self.models[1].predict_determined_input(inputs)
        mean_outputs = np.hstack((mean_outputs0, mean_outputs1))
        vars = np.hstack((var0, var1))
        input_dim = np.shape(inputs)[0]
        if input_dim == 1:
            mean_outputs = mean_outputs.reshape(-1)
            vars = vars.reshape(-1)
        return mean_outputs, vars
