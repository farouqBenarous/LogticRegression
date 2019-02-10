import numpy as np
import pandas as pd
from patsy import dmatrices
import warnings


def sigmoid(x):
    '''Sigmoid function of x.'''
    return 1 / (1 + np.exp(-x))


np.random.seed(0)  # set the seed

##Step 1 - Define model parameters (hyperparameters)

## algorithm settings

Convtol = 1e-8  # convergence tolerance

lam = None  # l2-regularization
max_iter = 20  # iterations


## data creation settings
# Covariance measures how two variables move together.
# It measures whether the two move in the same direction (a positive covariance)
# or in opposite directions (a negative covariance).
Cova = 0.95  # covariance between x and z
n_samples = 1000  # number of samples
sigma = 1  # variance of noise - how spread out is the data?

## model settings
beta_x, beta_z, beta_v = -4, .9, 1  # true beta coefficients
var_x, var_z, var_v = 1, 1, 4  # variances of inputs

## the model specification you want to fit
formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'

# ----------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------#


## Step 2 - Generate and organize our data

# The multivariate normal, multinormal or Gaussian distribution is a generalization of the one-dimensional normal
# distribution to higher dimensions. Such a distribution is specified by its mean and covariance matrix.
# so we generate values input values - (x, v, z) using normal distributions

# A probability distribution is a function that provides us the probabilities of all
# possible outcomes of a stochastic process.

# lets keep x and z closely related (height and weight)
x, z = np.random.multivariate_normal([0, 0], [[var_x, Cova], [Cova, var_z]], n_samples).T
# blood presure
v = np.random.normal(0, var_v, n_samples) ** 3

# create a pandas dataframe (easily parseable object for manipulation)
A = pd.DataFrame({'x': x, 'z': z, 'v': v})
# compute the log odds for our 3 independent variables
# using the sigmoid function
A['log_odds'] = sigmoid(A[['x', 'z', 'v']].dot([beta_x, beta_z, beta_v]) + sigma * np.random.normal(0, 1, n_samples))

# compute the probability sample from binomial distribution
# A binomial random variable is the number of successes x has in n repeated trials of a binomial experiment.
# The probability distribution of a binomial random variable is called a binomial distribution.
A['y'] = [np.random.binomial(1, p) for p in A.log_odds]

# create a dataframe that encompasses our input data, model formula, and outputs
y, X = dmatrices(formula, A, return_type='dataframe')

# print it
X.head(100)


# ----------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------#

# like dividing by zero (Wtff omgggggg universe collapses)
def catch_singularity(f):
    '''Silences LinAlg Errors and throws a warning instead.'''

    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Algorithm terminated - singular Hessian!')
            return args[0]

    return silencer


@catch_singularity
def newton_step(curr, X, lam=None):
    '''One naive step of Newton's Method'''

    # how to compute inverse? http://www.mathwarehouse.com/algebra/matrix/images/square-matrix/inverse-matrix.gif

    ## compute necessary objects
    # create probability matrix, miniminum 2 dimensions, tranpose (flip it)
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    # create weight matrix from it
    W = np.diag((p * (1 - p))[:, 0])
    # derive the hessian
    hessian = X.T.dot(W).dot(X)
    # derive the gradient
    grad = X.T.dot(y - p)

    ## regularization step (avoiding overfitting)
    if lam:
        # Return the least-squares solution to a linear matrix equation
        step, *_ = np.linalg.lstsq(hessian + lam * np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)

    ## update our 
    beta = curr + step

    return beta


@catch_singularity
def newton_step(curr, X, lam=None):
    '''One naive step of Newton's Method'''

    # how to compute inverse? http://www.mathwarehouse.com/algebra/matrix/images/square-matrix/inverse-matrix.gif

    ## compute necessary objects
    # create probability matrix, miniminum 2 dimensions, tranpose (flip it)
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    # create weight matrix from it
    W = np.diag((p * (1 - p))[:, 0])
    # derive the hessian
    hessian = X.T.dot(W).dot(X)
    # derive the gradient
    grad = X.T.dot(y - p)

    ## regularization step (avoiding overfitting)
    if lam:
        # Return the least-squares solution to a linear matrix equation
        step, *_ = np.linalg.lstsq(hessian + lam * np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)

    ## update our 
    beta = curr + step

    return beta


@catch_singularity
def alt_newton_step(curr, X, lam=None):
    '''One naive step of Newton's Method'''

    ## compute necessary objects
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    W = np.diag((p * (1 - p))[:, 0])
    hessian = X.T.dot(W).dot(X)
    grad = X.T.dot(y - p)

    ## regularization
    if lam:
        # Compute the inverse of a matrix.
        step = np.dot(np.linalg.inv(hessian + lam * np.eye(curr.shape[0])), grad)
    else:
        step = np.dot(np.linalg.inv(hessian), grad)

    ## update our weights
    beta = curr + step

    return beta


def check_coefs_convergence(beta_old, beta_new, Convtol, iters):
    '''Checks whether the coefficients have converged in the l-infinity norm.
    Returns True if they have converged, False otherwise.'''
    # calculate the change in the coefficients
    coef_change = np.abs(beta_old - beta_new)

    # if change hasn't reached the threshold and we have more iterations to go, keep training
    return not (np.any(coef_change > Convtol) & (iters < max_iter))


## initial olconditions
# initial coefficients (weight values), 2 copies, we'll update one
beta_old, beta = np.ones((len(X.columns), 1)), np.zeros((len(X.columns), 1))

# num iterations we've done so far
iter_count = 0
# have we reached convergence?
coefs_converged = False

# if we haven't reached convergence... (training step)
while not coefs_converged:
    # set the old coefficients to our current
    beta_old = beta
    # perform a single step of newton's optimization on our data, set our updated beta values
    beta = newton_step(beta, X, lam=lam)
    # increment the number of iterations
    iter_count += 1

    # check for convergence between our old and new beta values
    coefs_converged = check_coefs_convergence(beta_old, beta, Convtol, iter_count)

    print('Iterations : {}'.format(iter_count))
    print('Beta : {}'.format(beta))
