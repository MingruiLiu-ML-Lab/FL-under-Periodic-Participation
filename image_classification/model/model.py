import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import collections
from functools import reduce
from torch.autograd import Variable
from torch import nn
import copy

from model.optim import CorrectedSGD

LOSS_ACC_BATCH_SIZE = 128   # When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE


class Model():
    def __init__(self, rand_seed=None, learning_rate=0.001, model_name=None, device=None,
                 flatten_weight=False, pretrained_model_file = None, correction=None, architecture=None):
        super(Model, self).__init__()
        if device is None:
            raise Exception('Device not specified in Model()')
        if rand_seed is not None:
            torch.manual_seed(rand_seed)
        self.model = None
        self.loss_fn = None
        self.weights_key_list = None
        self.weights_size_list = None
        self.weights_num_list = None
        self.optimizer = None
        self.flatten_weight = flatten_weight
        self.learning_rate = learning_rate

        if model_name == 'ModelCNNMnist':
            from model.cnn_mnist import ModelCNNMnist
            self.model = ModelCNNMnist().to(device)
        elif model_name == 'ModelLinearMnist':
            self.model = nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28 * 28, 10)
            ).to(device)
        elif model_name == 'ModelCNNCifar10':
            from model.cnn_cifar10 import ModelCNNCifar10
            self.model = ModelCNNCifar10().to(device)
        elif model_name == 'ModelLinearCifar10':
            self.model = nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(3 * 32 * 32, 10)
            ).to(device)
        elif model_name == 'Model2LayerCifar10':
            if architecture is None:
                architecture = "8,5,2"
            else:
                split1 = architecture.find(",")
                split2 = architecture.find(",", split1+1)
                channels = int(architecture[:split1])
                kernel_size = int(architecture[split1+1:split2])
                stride = int(architecture[split2+1:])

            input_size = 32
            assert kernel_size % 2 == 1
            assert input_size % stride == 0
            padding = (kernel_size - 1) // 2
            activation_size = input_size // stride

            self.model = nn.Sequential(
                torch.nn.Conv2d(
                    3, channels, kernel_size=kernel_size, stride=stride, padding=padding
                ),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(channels * activation_size ** 2, 10)
            ).to(device)
        else:
            raise NotImplementedError

        if pretrained_model_file is not None:
            self.model.load_state_dict(torch.load(pretrained_model_file, map_location=device))

        self.correction = correction
        if self.correction is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = CorrectedSGD(self.model.parameters(), lr=learning_rate)

        self.model.to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        self._get_weight_info()

    def _get_weight_info(self):
        self.weights_key_list = []
        self.weights_size_list = []
        self.weights_num_list = []
        state = self.model.state_dict()
        for k, v in state.items():
            shape = list(v.size())
            self.weights_key_list.append(k)
            self.weights_size_list.append(shape)
            if len(shape) > 0:
                num_w = reduce(lambda x, y: x * y, shape)
            else:
                num_w=0
            self.weights_num_list.append(num_w)

    def get_weight_dimension(self):
        dim = sum(self.weights_num_list)
        return dim

    def get_weight(self):
        with torch.no_grad():
            state = self.model.state_dict()
            if self.flatten_weight:
                weight_flatten_tensor = torch.Tensor(sum(self.weights_num_list)).to(state[self.weights_key_list[0]].device)
                start_index = 0
                for i,[_, v] in zip(range(len(self.weights_num_list)), state.items()):
                    weight_flatten_tensor[start_index:start_index+self.weights_num_list[i]] = v.view(1, -1)
                    start_index += self.weights_num_list[i]

                return weight_flatten_tensor
            else:
                return copy.deepcopy(state)

    def get_diffable_weight(self):
        weight_list = []
        for p in self.model.parameters():
            weight_list.append(p.view(-1))
        return torch.cat(weight_list)

    def assign_weight(self, w):
        if self.flatten_weight:
            self.assign_flattened_weight(w)
        else:
            self.model.load_state_dict(w)

    def assign_flattened_weight(self, w):

        weight_dic = collections.OrderedDict()
        start_index = 0

        for i in range(len(self.weights_key_list)):
            sub_weight = w[start_index:start_index+self.weights_num_list[i]]
            if len(sub_weight) > 0:
                weight_dic[self.weights_key_list[i]] = sub_weight.view(self.weights_size_list[i])
            else:
                weight_dic[self.weights_key_list[i]] = torch.tensor(0)
            start_index += self.weights_num_list[i]
        self.model.load_state_dict(weight_dic)

    def _data_reshape(self, imgs, labels=None):
        if len(imgs.size()) < 3:
            x_image = imgs.view([-1, self.channels, self.img_size, self.img_size])
            if labels is not None:
                _, y_label = torch.max(labels.data, 1)  # From one-hot to number
            else:
                y_label = None
            return x_image, y_label
        else:
            return imgs, labels

    def accuracy(self, data_test_loader, w, device):
        if w is not None:
            self.assign_weight(w)

        self.model.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                images, labels = Variable(images).to(device), Variable(labels).to(device)
                output = self.model(images)
                avg_loss += self.loss_fn(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        avg_loss /= len(data_test_loader.dataset)
        acc = float(total_correct) / len(data_test_loader.dataset)

        return avg_loss.item(), acc

