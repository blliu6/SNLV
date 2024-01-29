from SumOfSquares import SOSProblem
from utils import Config
import logging as loger
import sympy as sp
from benchmarks.Examplers import Example
from utils.Convert import *
from constants import *
from functools import reduce
from itertools import product


class SOSValidator:
    def __init__(self, example: Example, config: Config, V):
        self.n = example.n
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(example.n)])
        self.local = zone_to_constraints(example.local, self.x)
        self.target = zone_to_constraints(example.target, self.x)
        self.center = example.target.center
        self.u = config.controller
        self.f = example.f
        self.V = V
        self.lie = self.get_Lyapunov()

        self.var_count = 0

    def solve(self):
        # \theta_i * s_i + V
        self.verify(self.construct_constraints(Constant.SUBSET_CONSTR))
        # -Lie - V - \sum {\phi_j * h_j}
        self.verify(self.construct_constraints(Constant.LL_CONSTR))
        # -V(x_0)
        self.verify(self.construct_constraints(Constant.NONEMPTY_CONSTR))

    def verify(self, expr):
        prob = SOSProblem()
        for e in expr:
            prob.add_sos_constraint(e, self.x)
        try:
            prob.solve(solver=Constant.SOLVER_TYPE)
        except:
            loger.error("solve failed.")

    def construct_constraints(self, constr_type, deg=2):

        if constr_type == Constant.SUBSET_CONSTR:
            return self._construct_subset_constraint(deg)

        if constr_type == Constant.LL_CONSTR:
            return self._construct_LL_constraint(deg)

        if constr_type == Constant.NONEMPTY_CONSTR:
            return self._construct_nonempty_constraint()

    def _construct_subset_constraint(self, deg=2):
        expr = []
        for i in range(len(self.target)):  # for target上整个半代数集.
            P, _, _ = self.polynomial(deg)
            expr.append(P)  # 需要保证乘子本身是SOS的
            expr.append(self.local[i] * P + self.V)  # \theta_i * s_i + V
        return expr

    def _construct_LL_constraint(self, deg=2):
        expr = []
        t = 0
        for i in range(len(self.local)):
            P, _, _ = self.polynomial(deg)
            expr.append(P)
            t += P * self.local[i]
        expr.append(-self.lie - self.V - t)

        return expr

    def _construct_nonempty_constraint(self):
        expr = sp.lambdify(self.x, self.V)
        return [-expr(*self.center)]

    def get_Lyapunov(self):
        diff = np.array([sp.diff(self.V, self.x[i]) for i in range(len(self.n))])
        f = np.array(self.f[i](x, self.u[i]) for i in range(self.n))
        lie = diff.T @ f

        return lie

    def polynomial(self, deg=2):  # Generating polynomials of degree n-ary deg.
        """

        :param deg: the polynomial's degree
        :return: a list, [polynomial, (optimal) parameter list, term list]
        """

        assert deg % 2 == 0

        if deg == 2 and self.n > 8:
            parameters = []
            terms = []
            poly = 0
            parameters.append(sp.symbols('parameter' + str(self.var_count)))
            self.var_count += 1
            poly += parameters[-1]
            terms.append(1)
            for i in range(self.n):
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i])
                poly += parameters[-1] * terms[-1]
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i] ** 2)
                poly += parameters[-1] * terms[-1]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    parameters.append(sp.symbols('parameter' + str(self.var_count)))
                    self.var_count += 1
                    terms.append(self.x[i] * self.x[j])
                    poly += parameters[-1] * terms[-1]
            return poly, parameters, terms
        else:
            parameters = []
            terms = []
            exponents = list(product(range(deg + 1), repeat=self.n))  # Generate all possible combinations of indices.
            exponents = [e for e in exponents if sum(e) <= deg]  # Remove items with a count greater than deg.
            poly = 0
            for e in exponents:  # Generate all items.
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(reduce(lambda a, b: a * b, [self.x[i] ** exp for i, exp in enumerate(e)]))
                poly += parameters[-1] * terms[-1]
            return poly, parameters, terms


if __name__ == "__main__":
    validator = SOSValidator()
    x = sp.symbols(['x{}'.format(i + 1) for i in range(2)])
    expr = x[0] ** 2 + 2 * x[0] + 1
    validator.verify([expr])
