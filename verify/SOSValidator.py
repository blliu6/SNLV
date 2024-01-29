from SumOfSquares import SOSProblem
from utils import Config
import logging as loger
import sympy as sp
from benchmarks.Examplers import Example


class SOSValidator:
    def __init__(self, example: Example):
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(example.n)])

    def verify(self, expr):
        prob = SOSProblem()
        for e in expr:
            prob.add_sos_constraint(e, self.x)
        try:
            prob.solve(solver=Config.SOLVER_TYPE)
        except:
            loger.error("solve failed.")

    def construct_constraints(self, constr_type):

        if constr_type == Config.SUBSET_CONSTR:
            pass

        if constr_type == Config.LL_CONSTR:
            pass

        if constr_type == Config.NONEMPTY_CONSTR:
            pass

    def _construct_subset_constraint(self):
        pass

    def _construct_LL_constraint(self):
        pass

    def _construct_nonempty_constraint(self):
        pass



if __name__ == "__main__":
    validator = SOSValidator()
    x = sp.symbols(['x{}'.format(i + 1) for i in range(2)])
    expr = x[0] ** 2 + 2 * x[0] + 1
    validator.verify([expr])
