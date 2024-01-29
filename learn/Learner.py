import torch
from utils.Config import Config
from learn.Net import Net


class Learner:
    def __init__(self, config: Config):
        self.net = Net(config)
        self.config = config

    def train(self, data):
        learn_loops = self.config.LEARNING_LOOPS
        margin = self.config.MARGIN
        slope = 1e-3
        relu6 = torch.nn.ReLU6()
        LR = self.config.LEARNING_RATE
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=LR)

        data_tensor = data

        for epoch in range(learn_loops):
            optimizer.zero_grad()

            v_t, v_y, v_grad, v_center = self.net(data_tensor)
            v_t, v_y, v_center = v_t[:, 0], v_y[:, 0], v_center

            weight = self.config.LOSS_WEIGHT

            accuracy = [0] * 3
            ###########
            # loss 1
            p = v_t
            accuracy[0] = sum(p > margin / 2).item() * 100 / len(v_t)

            loss_1 = weight[0] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()

            ###########
            # loss 2
            p = v_grad + v_y
            accuracy[1] = sum(p < -margin / 2).item() * 100 / len(v_y)

            loss_2 = weight[1] * (torch.relu(p + margin) - slope * relu6(-p - margin)).mean()
            ###########
            # loss 3
            p = v_center
            accuracy[2] = sum(p < -margin / 2).item() * 100 / len(v_center)
            loss_3 = weight[2] * (torch.relu(p + margin) - slope * relu6(-p - margin)).mean()
            ###########

            loss = loss_1 + loss_2 + loss_3
            # loss = loss_1 + loss_3
            # loss = loss_2
            result = True

            for e in accuracy:
                result = result and (e == 100)

            if epoch % (learn_loops // 10) == 0 or result:
                print(f'{epoch}->', end=' ')
                for i in range(len(accuracy)):
                    print(f'accuracy{i + 1}:{accuracy[i]}', end=', ')
                print(f'loss:{loss}')

            loss.backward()
            optimizer.step()
            if result:
                break
