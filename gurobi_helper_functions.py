from gurobipy import *
import pandas as pd
import numpy as np
import time

def calculate_asset_cvars(data, beta=0.95):
    """
    Helper function to calculate the Conditional Value at Risk-statistics for single asset
    portfolios (i.e., the case x_i = 1 and x_j = 0 for j =/= i , for all assets i).

    Parameters:
    data (pd.DataFrame): The scenario data based on which we calculate the CVar.

    beta (float): Confidence level parameter in the CVaR function. Specifies the range to where 
    we integrate the tail end of our loss distribution.

    Returns:

    asset_cvars (numpy.ndarray): The CVaR data for our individual assets.
    """

    #Number of assets, assumed to be columns in our data frame
    _, num_assets = data.shape
    asset_cvars = []

    values = -data.values

    for i in range(num_assets):
        asset_data = values[:,i] 
        p = np.quantile(asset_data,beta)
        asset_cvars.append(np.mean(asset_data[asset_data >= p]))

    return np.array(asset_cvars)

def calculate_portfolio_cvar(data, portfolio_vector, beta=0.95):
    """
    Helper function to calculate the Conditional Value at Risk-statistics for single asset
    portfolios (i.e., the case x_i = 1 and x_j = 0 for j =/= i , for all assets i).

    Parameters:
    data (pd.DataFrame): The scenario data based on which we calculate the CVar.

    portfolio_vector (pd.Series): Container holding the Gurobi variables x_i of an optimal portfolio.

    beta (float): Confidence level parameter in the CVaR function. Specifies the range to where 
    we integrate the tail end of our loss distribution.

    Returns:

    asset_cvars (numpy.ndarray): The CVaR data for our individual assets.
    """
    num_scen = data.shape[0]
    neg_values = -data.values
    x = np.array([v.x for v in portfolio_vector])

    asset_data = neg_values.dot(x)
    p = np.quantile(asset_data,beta)
    
    #array of values in the tail of the loss distribution
    extreme_values = asset_data[asset_data >= p]
    #array of indices where the sample quantile is taken as a data point
    quantile_taken = np.where(asset_data == p)[0]

    cutoff = int(len(extreme_values) - (1-beta)*num_scen)
    if cutoff > 0:
        inds = quantile_taken[:cutoff]
        asset_data[inds] = 0

    #cvar defined as mean of tail distribution
    return np.mean(extreme_values)

def calculate_portfolio_cvars(data, portfolio_matrix, beta=0.95):
    """
    Helper function to calculate the Conditional Value at Risk-statistic for a given matrix of portfolios.

    Parameters:
    data (pd.DataFrame): The scenario data based on which we calculate the CVar.

    portfolio_matrix (np.array): 2D np.array with columns being optimal portfolios.

    beta (float): Confidence level parameter in the CVaR function. Specifies the range to where 
    we integrate the tail end of our loss distribution.

    Returns:

    portfolio_cvars (numpy.ndarray): The CVaR data for our individual assets.
    """
    num_scen = data.shape[0]
    num_points = portfolio_matrix.shape[1]
    neg_values = -data.values
    portfolio_cvars = np.zeros(num_points)
    data_vecs = neg_values.dot(portfolio_matrix) #array, shape (50000, #portfolios)

    for i in range(num_points):
        asset_data = data_vecs[:,i]
        p = np.quantile(data_vecs[:,i],beta)

        #array of values in the tail of the loss distribution
        extreme_values = asset_data[asset_data >= p]
        #array of indices where the sample quantile is taken as a data point
        quantile_taken = np.where(asset_data == p)[0]
        
        cutoff = int(len(extreme_values) - (1-beta)*num_scen)
        if cutoff > 0:
            inds = quantile_taken[:cutoff]
            asset_data[inds] = 0

        portfolio_cvars[i] = np.mean(extreme_values)

    return portfolio_cvars


def test_performance(data, risk_measure="squared_risk", runs=10, num_points=50):
    """
    This routine is used to investigate how the performance of a model depends on the number of scenarios n.

    Parameters:
    data (pd.DataFrame): pandas DataFrame holding our scenario data.

    risk_measure (String): Name of the risk model we want to test. Either "squared_risk" or "cvar". 

    runs (integer): Number of times we run a model in succession. A larger number produces a better mean runtime value,
    but may require a longer runtime.

    num_points (integer): Number of points we want to measure. Each point is a number of scenarios that we will use to calculate our model.
    
    Output:
    points_range (np.ndarray): Vector with the scenario numbers for which runtime was tested.

    means: Mean runtimes for all tested scenario numbers.

    stddevs: Standard deviations for all tested scenario numbers.
    """

    if not isinstance(risk_measure, str):
        raise Exception("Expected a name string specifying the risk measure")
    
    if risk_measure != "squared risk" and risk_measure != "cvar":
        raise Exception("risk_measure argument must be set to either \"squared_risk\" or \"cvar\"")

    #convert both arguments to points to prevent issues
    int(runs)
    int(num_points)

    num_scen, num_assets = data.shape

    if num_points > num_scen:
        raise Exception("Number of measurement points cannot be greater than the number of scenarios")

    points_range = int((num_scen / num_points)) * np.arange(1,num_points+1) + (num_scen % num_points) - 1 #arrays start at zero

    means = []
    stddevs = []

    for i, num_rows in enumerate(points_range):
        asset_data = data[:num_rows]
        times = []

        if risk_measure == "squared_risk":
            model = build_squared_risk_model(asset_data)
        else:
            model = build_cvar_model(asset_data)

        #for loop for different runs 
        for k in range(runs):
            
            model.setParam('OutputFlag', 0)

            start = time.clock()
            model.optimize()
            end = time.clock()

            times.append(start-end)
            model.reset()

        mean = np.mean(times)
        std = np.std(times)

        means.append(mean)
        stddevs.append(std)
    
    return points_range, means, stddevs


def build_squared_risk_model(data):
    """
    Helper function that returns a Gurobi squared risk model built from risk data.

    Parameters:

    data (pd.DataFrame): pandas data frame that holds the asset/scenario data.

    Returns:
    model (Gurobi model variable): The model with constraints etc.

    x (pd.Series): Series object holding the asset variables.
    """
    m = Model('squared_risk')

    assets = data.columns

    x = pd.Series(m.addVars(assets, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS), index=assets)

    sigma = data.cov()
    squared_portfolio_risk = sigma.dot(x).dot(x)
    m.setObjective(squared_portfolio_risk, GRB.MINIMIZE)

    #portfolio constraint
    m.addConstr(x.sum() == 1, 'investment')
    m.update()

    return m, x

def build_cvar_model(data, beta=0.95):
    """
    Helper function that returns a Gurobi CVaR model built from risk data.
    
    Parameters:

    data (pd.DataFrame): pandas data frame that holds the asset/scenario data.

    beta (float): Quantile parameter for the conditional value at risk.

    Returns:
    model (Gurobi model variable): The model with variables, constraints etc.

    x (pd.Series): Series object holding the asset variables.
    """

    m = Model('portfolio')

    assets = data.columns

    # Add a variable for each asset
    x = pd.Series(m.addVars(assets, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS), index=assets)
    alpha = m.addVar(vtype=GRB.CONTINUOUS, name = "alpha") 

    m.update() 

    num_scen = len(data.index) #number of scenarios
    nu = 1 / ((1 - beta) * num_scen) #parameter in the objective function

    #Helper variables
    z = m.addVars(num_scen, lb=0.0, vtype=GRB.CONTINUOUS, name="z")
    m.update()

    ### Constraints:
    m.addConstr(x.sum() == 1, name='budget')
    asset_data = data.values

    z_constraints = -asset_data.dot(x) - np.repeat(LinExpr(alpha), num_scen)

    #nonnegativity constraints on the z_i
    m.addConstrs((z[i] >= z_constraints[i]) for i in range(num_scen))
    m.update()

    #Linear approximation of the CVaR
    cvar_approximation = alpha + nu * z.sum()
    m.setObjective(cvar_approximation, GRB.MINIMIZE)

    return m, x


def calculate_efficient_frontier(model, portfolio_vector, data, interval=None, num_points=100):
    """
    Helper function that calculates the efficient frontier generated by a risk model. Allocates variables and returns them.
    
    Parameters:
    model (Gurobi model variable): The model with variables, constraints etc.
    Assumes the asset variables are saved as a vector x.

    portfolio_vector (pd.Series): Container holding the asset variables (and also their values).

    data (pd.DataFrame object): Contains the asset scenario data.

    Interval (np.array): np.ndarray containing left and right bounds between which the frontier should
    be evaluated. 

    num_points (integer): Number of points to evaluate the efficient frontier on.

    Returns:

    frontier (numpy.ndarray): Array holding the efficient frontier values.

    return_values (numpy.ndarray): Array holding the expected return values for our optimal portfolios.

    optimal_portfolios (numpy.ndarray): 2D-array holding the optimal portfolios as columns.
    
    """

    mean_returns = data.mean()

    #optimize to find global optimum
    model.setParam('OutputFlag', 0)
    model.optimize()
    
    expected_return = mean_returns.dot(portfolio_vector)
    minrisk_return = expected_return.getValue()
    
    # Add (redundant) target return constraint
    target = model.addConstr(expected_return == minrisk_return, 'target')
    model.update()

    # Solve for efficient frontier by varying target return
    portfolio_vals = np.zeros((len(portfolio_vector),num_points))
    frontier = np.zeros(num_points)
    return_vals = np.zeros(num_points)

    #holds the model's objective function
    obj = model.getObjective()

    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    if interval is not None:
        assert(len(interval)==2), "the interval should contain exactly two points"
        assert(interval[0] >= min_ret), "Left bound should be larger than minimal return so the point is feasible"
        assert(interval[1] <= max_ret), "Right bound should be smaller than minimal return so the point is feasible"
        min_ret = interval[0]
        max_ret = interval[1]

    for i,r in enumerate(np.linspace(min_ret, max_ret, num_points)):
        target.rhs = r
        model.update()
        model.optimize()
        portfolio_vals[:,i] = np.array([v.x for v in portfolio_vector])
        frontier[i] = obj.getValue()
        return_vals[i] = r

    return frontier, return_vals, portfolio_vals

def calculate_squared_risk(data, portfolio_matrix):
    """
    Subroutine to calculate the squared risk values for a series of optimal portfolios.

    Inputs:

    data (pd.DataFrame): Asset data used in the simulations.

    portfolio_matrix (np.ndarray): Matrix with optimal portfolios as columns.

    Outputs:

    squared_risk_values (np.ndarray): Vector with squared risk values corresponding to the portfolios.
    """

    cov_mat = data.cov()

    return np.sum(portfolio_matrix * (cov_mat.dot(portfolio_matrix)), axis=0)    

def add_penalization(cvar_model, portfolio_vec, rho=None):
    """
    Add penalization to a Gurobi model for the Conditional Value at Risk. For best results, 
    models that are returned by the function build_cvar_model() should be passed as the cvar_model argument.

    Inputs:
    cvar_model (Gurobi.model object): Gurobi CVaR model without penalization
    rho (np.ndarray, length == 2): Regularization parameters.
    """

    if rho is None:
        return
    
    assert(len(rho)==2), "Two penalization constants must be passed"

    rho_1 = rho[0]
    rho_2 = rho[1]

    equal_weight_portfolio = np.ones(len(portfolio_vec)) / len(portfolio_vec)
    dist = portfolio_vec.values - equal_weight_portfolio
    squared_penalty = np.dot(dist,dist)

    #Helper variables
    abs_vals = cvar_model.addVars(len(portfolio_vec), lb=0.0, vtype=GRB.CONTINUOUS, name="abs_vals")
    abs_vals_placeholder = cvar_model.addVars(len(portfolio_vec),lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="abs_vals_placeholder")
    #have to add L1 penalty by dummy variables
    cvar_model.addConstrs((abs_vals[i] == abs_(abs_vals_placeholder[i])) for i in range(len(portfolio_vec)))
    cvar_model.addConstrs((abs_vals_placeholder[i] == dist[i] for i in range(len(portfolio_vec))))
    cvar_model.update()

    #Query CVaR objective function
    obj = cvar_model.getObjective()
    cvar_penalty_approximation = obj + rho_1 * abs_vals.sum() + rho_2 * squared_penalty

    cvar_model.setObjective(cvar_penalty_approximation, GRB.MINIMIZE)