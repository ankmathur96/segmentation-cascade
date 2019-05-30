from __future__ import print_function
import os
import argparse
import torch
import shutil
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv
from torchvision import datasets, transforms
from torch.utils import data
from video_utils import get_all_images_for_scene
class MovingDataset(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.scene_to_label = {1:0, 3:1, 4:2, 5:3}
        self.scene_indices = [1,3,4,5]
        self.num_classes = 4
        self.print_every = 5
        self.num_train_steps = 1000
        self.batch_size = 8
        self.image_paths, self.labels = self.load_data()

    def load_data(self):
        print("Loading data ... ")
        image_paths, labels = [], []
        for scene_idx in self.scene_indices:
            images_from_scene = get_all_images_for_scene(scene_idx, root=self.root)
            image_paths.extend(images_from_scene)
            labels.extend([self.scene_to_label[scene_idx] for _ in range(len(images_from_scene))])
        print(image_paths)
        print("finished.")
        return image_paths, labels

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.labels)


BASE_PATH = 'saved_models/'
PAD = 2
class Net(nn.Module):
    def __init__(self, n_filters_1, n_filters_2, linear_1_units, filter_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filters_1, kernel_size=filter_size, padding=PAD)
        self.conv2 = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=filter_size, padding=PAD)
        padded_filter_size = filter_size + PAD
        flattened_units = n_filters_2 * padded_filter_size * padded_filter_size
        self.fc1 = nn.Linear(flattened_units, linear_1_units)
        self.fc2 = nn.Linear(linear_1_units, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.log_softmax_result = None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        print(x.size())
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        self.log_softmax_result = self.log_softmax(x)
        return self.log_softmax_result

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        if batch_idx % 100 == 0:
            losses.append((epoch + batch_idx / len(train_loader), loss.item()))
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return losses

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy

def save_checkpoint(state, is_best, save_path):
    filename = save_path + 'checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + 'model_best.pth.tar')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--fc3-units', type=int, default=1024)
    parser.add_argument('--c1-filters', type=int, default=32)
    parser.add_argument('--c2-filters', type=int, default=64)
    parser.add_argument('--filter-size', type=int, default=5)
    parser.add_argument('--checkpoint', action='store_true', default=False, help='Save the checkpoints?')
    parser.add_argument('--output-losses', action='store_true', default=False, help='Output the losses?')
    args = parser.parse_args()
    n_filters_1 = args.c1_filters
    n_filters_2 = args.c2_filters
    filter_size = args.filter_size
    linear_1_units = args.fc3_units
    model_name = 'lenet_' + str(n_filters_1) + '_' + str(n_filters_2) + '_' + str(linear_1_units)
    save_path = BASE_PATH + model_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Created save directory at:', save_path)
    else:
        print('Overwriting files at:', save_path)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        MovingDataset('/Users/ankitmathur/datasets/project/', transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net(n_filters_1, n_filters_2, linear_1_units, filter_size).to(device)
    print('model defined.')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    best_accuracy = 0
    plot_loss_values = []
    plot_accuracy_values = []
    for epoch in range(1, args.epochs + 1):
        print('epoch', epoch)
        loss_values = train(args, model, device, train_loader, optimizer, epoch)
        plot_loss_values.extend(loss_values)
        accuracy = test(args, model, device, train_loader)
        plot_accuracy_values.append((epoch, accuracy))
        if accuracy > best_accuracy:
            is_best = True
            best_accuracy = accuracy
        else:
            is_best = False
        if args.checkpoint:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path)
    if args.output_losses:
        np.savez('losses_' + model_name + '.npz', loss_vals=np.array(plot_loss_values), accuracy_vals=np.array(plot_accuracy_values))


if __name__ == '__main__':
    main()