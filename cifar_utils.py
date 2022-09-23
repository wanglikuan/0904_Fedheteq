import torch
import torchvision
from PIL import Image
import torch.nn as nn
import random
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

#from generate_iid_new_origin_dirichlet import rearrange_data_by_class
#from torchvision.dataset.vision import VisionDataset
#from torchvision.dataset.utils import check_integrity, download_and_extract_archive
random.seed(42)
np.random.seed(42)

class CIFAR10Instance(torchvision.datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10Instance, self).__init__(root=root,
                                                           train=train,
                                                           transform=transform,
                                                           target_transform=target_transform)

        self.origin_data = []        
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.origin_data.append(entry['data'])

    def _save_new_dataset(self, dataset_x,dataset_y, id):
        temp_shuffle_dataset = list(zip(dataset_x,dataset_y))
        random.shuffle(temp_shuffle_dataset)
        dataset_x[:], dataset_y[:] = zip(*temp_shuffle_dataset)
        dataset={'data': dataset_x, 'labels': dataset_y}
        file_path = os.path.join(self.root, 'generated_dataset', 'iid_data_client{}'.format(id))
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        return None

    def _split_iid_train_data(self, client_num):
        data_by_class = self.rearrange_data_by_class(
            self.origin_data.detach(),
            self.targets.cpu().detach().numpy(),
            len(self.classes)
        )
        proportions = [ 1000, 2000, 3000, 4000]     
        idx_batch=[{} for _ in range(client_num)] 
        for l in range(len(self.classes)):
            idx_l = [i for i in range(len(data_by_class[l]))]  
        for u, new_idx in enumerate(np.split(idx_l, proportions)):  # np.split(idx_l, proportions
            idx_batch[u][l] = new_idx.tolist()
        X = [[] for _ in range(client_num)]
        y = [[] for _ in range(client_num)]
        print("processing users...")
        for u, user_idx_batch in enumerate(idx_batch):
            for l, indices in user_idx_batch.items():
                if len(indices) == 0: continue
                X[u] += data_by_class[l][indices].tolist()
                y[u] += (l * np.ones(len(indices))).tolist()
        
        for u in range(client_num):
            self._save_new_dataset(X[u], y[u], u)

        return None

    def rearrange_data_by_class(self, data, targets, n_class):
        new_data = []
        for i in trange(n_class):
            idx = targets == i
            new_data.append(data[idx])
        return new_data

    def _check_integrity(self):
        return True

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


    def __getitem__(self, index):
        #if self.train:
        #    img, target = self.data[index], self.targets[index]
        # else:
        image, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if self.transform is not None:
            img = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class CIFAR100Instance(CIFAR10Instance):
    """CIFAR100Instance Dataset.
    This is a subclass of the `CIFAR10Instance` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def kNN(net, trainloader, testloader, K, sigma=0.1, dim=128,use_pca=False):
    net.eval()
    print('start knn...')
    print('start trainlabels...')
    # this part is ugly but made to be backwards-compatible. there was a change in cifar dataset's structure.
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]) # .cuda()
    elif hasattr(trainloader.dataset, 'indices'):
        trainLabels = torch.LongTensor([k for path,k in trainloader.dataset.dataset.dt.imgs])[trainloader.dataset.indices]
    elif hasattr(trainloader.dataset, 'train_labels'):
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()  # .cuda()
    if hasattr(trainloader.dataset, 'dt'):
        if hasattr(trainloader.dataset.dt, 'targets'):
            trainLabels = torch.LongTensor(trainloader.dataset.dt.targets) # .cuda()
        else: #  hasattr(trainloader.dataset.dt, 'imgs'):
            trainLabels = torch.LongTensor([k for path,k in trainloader.dataset.dt.imgs]) # .cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda() # .cuda()
        #tmp_list = []
        #for (x, y, i) in trainloader.dataset:
        #    tmp_list.append(y)
        #trainLabels = torch.LongTensor(tmp_list) # .cuda()
    C = trainLabels.max() + 1
    print('end C')
    print('start transform...')
    #if hasattr(trainloader.dataset, 'transform'):
    #    transform_bak = trainloader.dataset.transform
    #    trainloader.dataset.transform = testloader.dataset.transform
    #elif hasattr(trainloader.dataset.dataset.dt, 'transform'):
    #    transform_bak = trainloader.dataset.dataset.dt.transform
    #    trainloader.dataset.dataset.dt.transform = testloader.dataset.dt.transform
    #else:
    #    transform_bak = trainloader.dataset.dt.transform
    #    trainloader.dataset.dt.transform = testloader.dataset.dt.transform

    print('end transform')
    print('start temploader...')

    temploader = torch.utils.data.DataLoader(trainloader.dataset,
                                             batch_size=64, num_workers=1)
    if hasattr(trainloader.dataset, 'indices'):
        LEN = len(trainloader.dataset.indices)
    else:
        LEN = len(trainloader.dataset)
    trainFeatures = torch.zeros((dim, LEN))  # , device='cuda:0')
    normalize = Normalize()
    for batch_idx, (inputs, targets, _) in enumerate(temploader):
        batchSize = inputs.size(0)
        inputs = inputs.cuda()
        features = net(inputs)
        if not use_pca:
            features = normalize(features)
        trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t().cpu()

    print('end temploader')
    print('start trainlabels...')

    if hasattr(temploader.dataset, 'imgs'):
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda() # .cuda()
    elif hasattr(temploader.dataset, 'indices'):
        trainLabels = torch.LongTensor([k for path,k in temploader.dataset.dataset.dt.imgs])[temploader.dataset.indices]
    elif hasattr(temploader.dataset, 'train_labels'):
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda() # .cuda()
    elif hasattr(temploader.dataset, 'targets'):
        trainLabels = torch.LongTensor(temploader.dataset.targets).cuda() # .cuda()
    #elif hasattr(temploader.dataset.dt, 'imgs'):
    #    trainLabels = torch.LongTensor([k for path,k in temploader.dataset.dt.imgs]) #.cuda()
    #elif hasattr(temploader.dataset.dt, 'targets'):
    #    trainLabels = torch.LongTensor(temploader.dataset.dt.targets) #.cuda()
    else:
        trainLabels = torch.LongTensor(temploader.dataset.labels).cuda() #.cuda()
        #tmp_list = []
        #for (x, y, i) in temploader.dataset:
        #    tmp_list.append(y)
        #trainLabels = torch.LongTensor(tmp_list) # .cuda()        

    #trainLabels = trainLabels.cpu()
    trainLabels = trainLabels.cuda()
    print('end trainlabels')
    print('start transform...')

    #if hasattr(trainloader.dataset, 'transform'):
    #    trainloader.dataset.transform = transform_bak
    #elif hasattr(trainloader.dataset, 'indices'):
    #    trainloader.dataset.dataset.dt.transform = transform_bak
    #else:
    #    trainloader.dataset.dt.transform = transform_bak

    print('finish prepare')

    if use_pca:
        comps = 128
        print('doing PCA with %s components'%comps, end=' ')
        from sklearn.decomposition import PCA
        pca = PCA(n_components=comps, whiten=False)
        trainFeatures = pca.fit_transform(trainFeatures.numpy().T)
        trainFeatures = torch.Tensor(trainFeatures)
        trainFeatures = normalize(trainFeatures).t()
        print('..done')
    def eval_k_s(K_,sigma_):
        total = 0
        top1 = 0.
        top5 = 0.

        with torch.no_grad():
            retrieval_one_hot = torch.zeros(K_, C).cuda() # .cuda()
            for batch_idx, (inputs, targets, _) in enumerate(testloader):
                print('eval_k_s batch index:  ', batch_idx)
                targets = targets # .cuda(async=True) # or without async for py3.7
                inputs = inputs.cuda()
                batchSize = inputs.size(0)
                features = net(inputs)
                print('finish features = net(inputs)...')
                if use_pca:
                    features = pca.transform(features.cpu().numpy())
                    features = torch.Tensor(features).cuda()
                features = normalize(features).cpu()

                print('finish normalize(features).cpu()')
                print('features.shape:  ',features.shape,'    trainFeatures.shape:  ', trainFeatures.shape)

                half_features = features.cuda()
                half_trainFeatures = trainFeatures.cuda()

                #print('PS_pre.type:  ', features.type())
                #print('trainFeatures:  ', trainFeatures.type())
                #dist = torch.matmul(features, trainFeatures)
                dist = torch.matmul(half_features, half_trainFeatures)

                print('finish torch.mm(features, trainFeatures)')

                yd, yi = dist.topk(K_, dim=1, largest=True, sorted=True)
                candidates = trainLabels.view(1, -1).expand(batchSize, -1).cuda()
                retrieval = torch.gather(candidates, 1, yi)

                print('finish torch.gather(candidates, 1, yi)...')

                retrieval_one_hot.resize_(batchSize * K_, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1).cuda(), 1)
                yd_transform = yd.clone().div_(sigma_).exp_()

                print('finish yd.clone().div_(sigma_).exp_()...')

                probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C),
                                            yd_transform.view(batchSize, -1, 1)),
                                  1)
                _, predictions = probs.sort(1, True)

                # Find which predictions match the target
                correct = predictions.eq(targets.data.view(-1, 1).cuda())

                top1 = top1 + correct.narrow(1, 0, 1).sum().item()
                top5 = top5 + correct.narrow(1, 0, 5).sum().item()

                total += targets.size(0)

        print(f"{K_}-NN,s={sigma_}: TOP1: ", top1 * 100. / total)
        return top1 / total

    if isinstance(K, list):
        res = []
        for K_ in K:
            for sigma_ in sigma:
                res.append(eval_k_s(K_, sigma_))
        return res
    else:
        print('perform one eval_k_s...')
        res = eval_k_s(K, sigma)
        return res
