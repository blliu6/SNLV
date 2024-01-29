import timeit
import os
import torch
import numpy as np
from utils.Config import Config
from learn.Generate_data import Data
from learn.Learner import Learner
from verify.SOSValidator import SOSValidator


class Cegis:
    def __init__(self, config: Config):
        self.config = config
        self.max_cegis_iter = config.max_iter

    def solve(self):
        import mosek

        with mosek.Env() as env:
            with env.Task() as task:
                task.optimize()
                print('Mosek can be used normally.')

        data = Data(self.config).generate_data()
        learner = Learner(self.config)
        # counter = CounterExampleFinder(self.config)

        t_learn = 0
        t_cex = 0
        t_sos = 0
        vis_sos = False
        barrier = None
        for i in range(self.max_cegis_iter):
            print(f'iter:{i + 1}')
            t1 = timeit.default_timer()
            learner.train(data)
            t2 = timeit.default_timer()
            t_learn += t2 - t1

            barrier = learner.net.get_barriers()
            print(f'V(x) = {barrier}')
            # for poly in barrier:
            #     print(poly)
            validator = SOSValidator(self.config.EXAMPLE, self.config, barrier)
            t3 = timeit.default_timer()
            vis = validator.solve()

            if vis:
                print('All SOS verification passed!')
                print(f'V = {barrier}')
                print(f'Total number of iterations:{i + 1}')
                vis_sos = True
                t4 = timeit.default_timer()
                t_sos += t4 - t3
                break

            t4 = timeit.default_timer()
            t_sos += t4 - t3

            # t5 = timeit.default_timer()
            # if self.config.example.continuous:
            #     res = counter.find_counterexample_for_continuous(state, barrier)
            # else:
            #     res = counter.find_counterexample(state, barrier)
            # t6 = timeit.default_timer()
            # t_cex += t6 - t5
            #
            # data = self.merge_data(data, res)

            # if self.config.example.continuous:
            #     draw = Draw(self.config.example, barrier[0])
            #     draw.draw_continuous()

        end = timeit.default_timer()
        if vis_sos:
            print('Total learning time:{}'.format(t_learn))
            print('Total counter-examples generating time:{}'.format(t_cex))
            print('Total sos verifying time:{}'.format(t_sos))
            # self.save_model(learner.net, barrier)
            # if self.config.example.n == 2:
            #     if self.config.example.continuous:
            #         draw = Draw(self.config.example, barrier[0])
            #         draw.draw_continuous()
            #         draw.draw_3d_continuous()
            #     else:
            #         draw = Draw(self.config.example, barrier[0], barrier[1])
            #         draw.draw()
            #         draw.draw_3d()
        else:
            print('Failed')
        return end

    def merge_data(self, data, add_res):
        res = []
        ans = 0
        for x, y in zip(data, add_res):
            if len(y) > 0:
                ans = ans + len(y)
                y = torch.Tensor(np.array(y))
                res.append(torch.cat([x, y], dim=0).detach())
            else:
                res.append(x)
        print(f'Add {ans} counterexamples!')
        return tuple(res)

    def save_model(self, net, barrier):
        if not os.path.exists('../model'):
            os.mkdir('../model')
        torch.save(net.state_dict(), f'../model/{self.config.example.name}.pt')

        if not os.path.exists('../result'):
            os.mkdir('../result')

        with open(f'../result/{self.config.example.name}.txt', 'w') as f:
            if self.config.example.continuous:
                f.write(f'B={barrier[0]}')
            else:
                f.write(f'B1={barrier[0]}\n')
                f.write(f'B2={barrier[1]}')
