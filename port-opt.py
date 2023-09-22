from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.result import QuasiDistribution
from qiskit_aer.primitives import Sampler
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import matplotlib.pyplot as plt
import datetime
from qiskit_finance.data_providers import YahooDataProvider
from qiskit_finance.exceptions import QiskitFinanceError
import pandas as pd
from qiskit.utils import algorithm_globals
from scipy.stats import norm


#This functions calculates the result of the optimization and prints it

def print_result(result):
    selection = result.x
    value = result.fval
    #print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = portfolio.to_quadratic_program().objective.evaluate(x)
        #print("%10s\t%.4f\t\t%.4f" % (x, value, v))

# set number of assets (= number of qubits)
num_assets = 4
seed = 123
data_csv = pd.read_excel('holdings-daily-us-en-spy.xlsx').dropna()
Name_list = data_csv["Ticker"].values.tolist()
#print(Name_list)
for i,r in enumerate (Name_list):
    asset_list = Name_list[i:i+4]
    if len(asset_list) < 4:
        break
    else:
        #print(asset_list)
        # Generate expected return and covariance matrix from (random) time-series
        try:
            cnt = 0
            data = YahooDataProvider(
                tickers=asset_list,
                start=datetime.datetime(2023, 1, 1),
                end=datetime.datetime(2023, 9, 5),
            )
            data.run()
            #for (cnt, s) in enumerate(data._tickers):
                #plt.plot(data._data[cnt], label=s)
            #plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
            #plt.xticks(rotation=90)
            #plt.show()
        except QiskitFinanceError as ex:
            data = None
            print(ex)

        mu = data.get_period_return_mean_vector()
        sigma = data.get_period_return_covariance_matrix()
        
        
        if mu is None or sigma is None:
            print("Unable to compute, please check your input.")
            exit(0)
        # Plot between -10 and 10 with .001 steps.
        #plot normal distribution with mean 0 and standard deviation 1
        #plt.plot(mu, norm.pdf(mu, 0, 1))
        
        # plot sigma
        #plt.imshow(mu)
        #plt.show()
        
        q = 0.5  # set risk factor
        budget = num_assets // 2  # set budget
        penalty = num_assets  # set parameter to scale the budget penalty term
        #input("Press Enter to continue...")
        portfolio = PortfolioOptimization(
            expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
        )
        
        qp = portfolio.to_quadratic_program()
        qp


        exact_mes = NumPyMinimumEigensolver()
        exact_eigensolver = MinimumEigenOptimizer(exact_mes)

        result_classic = exact_eigensolver.solve(qp)

        print_result(result_classic)

        

        algorithm_globals.random_seed = 1234

        cobyla = COBYLA()
        cobyla.set_options(maxiter=500)
        ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
        vqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla)
        vqe = MinimumEigenOptimizer(vqe_mes)
        result_vqe = vqe.solve(qp)

        print_result(result_vqe)

        algorithm_globals.random_seed = 1234

        cobyla = COBYLA()
        cobyla.set_options(maxiter=250)
        qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=3)
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        result_qaoa = qaoa.solve(qp)
        print_result(result_qaoa)

        print ("-----------------")
        print (data._tickers)
        print("Optimal: selection Classic{}, value {:.4f}".format(result_classic.x, result_classic.fval))
        print("Optimal: selection VQE{}, value {:.4f}".format(result_vqe.x, result_vqe.fval))
        print("Optimal: selection QAOA{}, value {:.4f}".format(result_qaoa.x, result_qaoa.fval))
        print ("-----------------")
        output = pd.DataFrame([result_classic.x, result_vqe.x, result_qaoa.x], columns=data._tickers)
        output.to_csv('output.csv', mode='a', header=True)
        data = None
        portfolio = None
        qp.clear()
        #plt.close('all')
        result = None
        

