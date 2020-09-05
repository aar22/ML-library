import numpy as np

class LinearRegression:
    """
    Trains Linear Regression model and predicts output
    Parameters:
    -----------
    alpha: float
        The learning rate to be used for gradient descent algorithm
    iterations: int
        The number of iterations to run gardient descent for
    """
    def __init__(self, alpha=.01, iterations=1000):
        self.alpha = alpha
        self.iterations = iterations
        self.theta = None
    
    def _calculate_cost(self, X, y):
        """
        Computes means square error cost of using the linear regression model
        defined by parameters theta for output vector y and feature array X

        Parameters:
        X : array, shape(m, n+1)
        The input features where m is number of samples and n is number of features

        y : array, shape(m,)
        The output value for each sample

        theta : array, shape(n+1,)
        The parameters of the linear regression model or weight 
        associated with each feature

        Output:
        J : float
        Cost computed
        grad : array
        A vector of shape (n, ) which is the gradient of the cost
        function with respect to theta
        """
        num_samples = y.size
        predicted_y = np.dot(X,self.theta)
        J = np.square(predicted_y - y).sum()/(2*num_samples)
        grad = (1/num_samples)*np.dot(X.T,(predicted_y-y))
        return J, grad

    def fit_lin_reg(self, X, y):
        # initialize parameters with random values
        self.theta = np.zeros(X.shape[1],)
        cost_hist = []
        for iter in range(self.iterations):
            J, grad = self._calculate_cost(X, y)
            cost_hist.append(J)
            self.theta -= self.alpha*grad
        return cost_hist
        
    def predict(self, X):
        pred_y = X.dot(self.theta)
        return pred_y
