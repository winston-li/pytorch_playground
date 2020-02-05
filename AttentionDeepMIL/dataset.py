"""Pytorch dataset object that loads MNIST dataset as bags."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage.io import imsave

MNIST_TRAIN_SIZE = 60000
MNIST_TEST_SIZE = 10000
DIGIT_DIM = 28


class MnistBags(Dataset):
    def __init__(self,
                 target_number=9,
                 min_target_count=1,
                 mean_bag_length=15,
                 var_bag_length=5,
                 scale=10,
                 num_bag=1000,
                 seed=1,
                 train=True):
        self.target_number = target_number
        self.min_target_count = min_target_count
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.scale = scale
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = MNIST_TRAIN_SIZE
        self.num_in_test = MNIST_TEST_SIZE

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def create_random_img(self, bag):
        def _check(y, x, existing):
            for ty, tx in existing:
                if (abs(tx - x) < DIGIT_DIM) and (abs(ty - y) < DIGIT_DIM):
                    return False
            return True

        w = h = DIGIT_DIM * self.scale
        s = DIGIT_DIM
        pos = []
        while len(pos) < bag.size(0):
            x = random.randint(0, w - s)
            y = random.randint(0, h - s)
            if _check(y, x, pos):
                pos.append((y, x))
        img = np.zeros((1, h, w))
        for p, digit in zip(pos, bag):
            img[0, p[0]:p[0] + s, p[1]:p[1] + s] = digit[0]
        return img.astype(np.float32)

    def create_grid_img(self, bag):
        pos = [(y, x) for y in range(self.scale) for x in range(self.scale)]
        random.shuffle(pos)
        pick = pos[:bag.size(0)]
        img = np.zeros((1, DIGIT_DIM * self.scale, DIGIT_DIM * self.scale))
        for p, digit in zip(pick, bag):
            row, col = p[0], p[1]
            img[0, row * DIGIT_DIM:(row + 1) * DIGIT_DIM, col *
                DIGIT_DIM:(col + 1) * DIGIT_DIM] = digit[0]
        return img.astype(np.float32)

    def _create_bags(self):
        if self.train:
            loader = DataLoader(datasets.MNIST('../datasets',
                                               train=True,
                                               download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                               ])),
                                batch_size=self.num_in_train,
                                shuffle=False)
        else:
            loader = DataLoader(datasets.MNIST('../datasets',
                                               train=False,
                                               download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                               ])),
                                batch_size=self.num_in_test,
                                shuffle=False)
        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

            target_index = [
                i for i, label in enumerate(all_labels)
                if label == self.target_number
            ]
            others_index = [
                i for i, label in enumerate(all_labels)
                if label != self.target_number
            ]
            print(
                f'all_imgs = {all_imgs.shape}, all_labels = {all_labels.shape}'
            )
            print(
                f'target_index size = {len(target_index)}, others_index size = {len(others_index)}'
            )

        bags_list = []
        labels_list = []

        for _ in range(self.num_bag):
            bag_length = np.int(
                self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < self.min_target_count:
                bag_length = self.min_target_count

            if random.random() > 0.5:  # create negative bag
                if random.random() > 0.5:  # no target
                    indices = random.choices(others_index, k=bag_length)
                    labels_in_bag = all_labels[indices]
                    bag_label = sum(labels_in_bag ==
                                    self.target_number) > self.min_target_count
                    bags_list.append(all_imgs[indices])
                    labels_list.append(bag_label)
                else:  # with target, but less than min count
                    tgt_count = random.randrange(
                        1, self.min_target_count
                    ) if self.min_target_count > 1 else 1
                    indices = []
                    if tgt_count > 0:
                        picks = random.choices(target_index, k=tgt_count)
                        indices.extend(picks)
                    picks = random.choices(others_index,
                                           k=bag_length - tgt_count)
                    indices.extend(picks)
                    labels_in_bag = all_labels[indices]
                    bag_label = sum(labels_in_bag ==
                                    self.target_number) > self.min_target_count
                    bags_list.append(all_imgs[indices])
                    labels_list.append(bag_label)
            else:  # create positive bag
                indices = []
                tgt_count = random.randrange(
                    self.min_target_count, bag_length
                ) if bag_length > self.min_target_count else self.min_target_count
                picks = random.choices(target_index, k=tgt_count)
                indices.extend(picks)
                picks = random.choices(others_index, k=bag_length - tgt_count)
                indices.extend(picks)
                labels_in_bag = all_labels[indices]
                bag_label = sum(labels_in_bag ==
                                self.target_number) > self.min_target_count
                bags_list.append(all_imgs[indices])
                labels_list.append(bag_label)
        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            # img = self.create_grid_img(bag)
            img = self.create_random_img(bag)
            label = self.train_labels_list[index]
        else:
            bag = self.test_bags_list[index]
            # img = self.create_grid_img(bag)
            img = self.create_random_img(bag)
            label = self.test_labels_list[index]
        return img, label


if __name__ == "__main__":
    train_loader = DataLoader(MnistBags(target_number=9,
                                        min_target_count=3,
                                        mean_bag_length=15,
                                        var_bag_length=5,
                                        scale=10,
                                        num_bag=6000,
                                        seed=1,
                                        train=True),
                              batch_size=1,
                              shuffle=True)

    test_loader = DataLoader(MnistBags(target_number=9,
                                       min_target_count=3,
                                       mean_bag_length=15,
                                       var_bag_length=5,
                                       scale=10,
                                       num_bag=1000,
                                       seed=1,
                                       train=False),
                             batch_size=1,
                             shuffle=False)

    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        if batch_idx <= 2:
            img = bag.cpu().numpy().squeeze()
            imsave(f'test{batch_idx}_{label.numpy()}.png', img)
        mnist_bags_train += label.numpy()[0]
    print('Number positive train bags: {}/{}\n'.format(mnist_bags_train,
                                                       len(train_loader)))

    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        mnist_bags_test += label.numpy()[0]
    print('Number positive test bags: {}/{}\n'.format(mnist_bags_test,
                                                      len(test_loader)))
