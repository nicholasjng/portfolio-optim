# portfolio-optim
University project on Portfolio Optimization using Linear Programming and Gurobi. 
The objective was to investigate various algorithms for optimizing a portfolio of financial assets with respect to a risk measure.
My part was optimizing the [Conditional Value-at-Risk (CVaR)](https://en.wikipedia.org/wiki/Expected_shortfall), also called Expected Shortfall, using Linear Programming. 

Running this notebook requires a valid Gurobi license and the Gurobipy interface. For installing Gurobipy via Anaconda, use

```
conda config --add channels http://conda.anaconda.org/gurobi

conda install gurobi
```
