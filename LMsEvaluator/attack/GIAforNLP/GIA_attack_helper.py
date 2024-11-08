import torch
from ..base_attack import BaseAttack
from utils.my_exception import print_red


class AttackModel(BaseAttack):
    def __init__(self, model, tokenizer, train_iter, distance_func, gradients, config_parser):
        super().__init__(config_parser)
        self.device = torch.device(
            'mps' if (config_parser['general']['useGpu'] and torch.backends.mps.is_available()) else 'cpu')
        self.model = model
        self.tokenizer = tokenizer
        self.train_iter = train_iter
        self.gradients = gradients

        if distance_func == "l2":
            self.distance_func = self.l2_loss
        elif distance_func == "cos":
            self.distance_func = self.cos_loss
        else:
            print_red("Please check the attackArgs.distanceFunc config in config.yaml.")
            self.distance_func = self.cos_loss
        # self.true_grads = torch.autograd.grad(outs.loss, model.parameters(), create_graph=False, allow_unused=True)

    def l2_loss(self, grads1, grads2):
        l2 = 0
        for g1, g2 in zip(grads1, grads2):
            if (g1 is not None) and (g2 is not None):
                l2 += (g1 - g2).square().sum()
        return l2

    def cos_loss(self, grads1, grads2):
        cos = 0
        n_g = 0
        for g1, g2 in zip(grads1, grads2):
            if (g1 is not None) and (g2 is not None):
                cos += 1.0 - (g1 * g2).sum() / (g1.view(-1).norm(p=2) * g2.view(-1).norm(p=2))
                n_g += 1
        cos /= n_g
        # temp = torch.nn.functional.cosine_similarity(torch.tensor(grads1), torch.tensor(grads2))
        # print("my_cos_loss:")
        # print(cos)
        # print("torch_cos_loss:")
        # print(temp)
        return cos
