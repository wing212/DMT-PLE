import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from apex import amp
from collections import Counter
import time
from functional import crop

# Base directories
base_voc = '../../voc_seg_deeplab/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
base_city = '../../../dataset/cityscapes'

# Common parameters
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
coco_mean = [104.008, 116.669, 122.675]  # BGR
coco_std = [1.0, 1.0, 1.0]
city_mean = [73.15835918458554, 82.90891773640608, 72.39239908619095]
voc_mean = [116.52101153914718, 111.3575037556515, 102.92616541705553]

# Here 'training resize min' is also the final training crop size as RandomResize & RandomCrop are used together
# For PASCAL VOC 2012
sizes_voc = [(321, 321), (505, 505), (505, 505)]  # training resize min/training resize max/testing label size
num_classes_voc = 21
colors_voc = [[0, 0, 0],
              [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [255, 255, 255]]
categories_voc = ['Background',
                  'Aeroplane', 'Bicycle', 'Bird', 'Boat',
                  'Bottle', 'Bus', 'Car', 'Cat',
                  'Chair', 'Cow', 'Diningtable', 'Dog',
                  'Horse', 'Motorbike', 'Person', 'Pottedplant',
                  'Sheep', 'Sofa', 'Train', 'Tvmonitor']

# For cityscapes (19 classes, ignore as black, no such thing as background)
sizes_city = [(256, 512), (512, 1024), (512, 1024)]  # training resize min/training resize max/testing label size
num_classes_city = 19
colors_city = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
    [0, 0, 0]]
categories_city = [
    'road', 'sidewalk', 'building', 'wall',
    'fence', 'pole', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle']
label_id_map_city = [255, 255, 255, 255, 255, 255, 255,
                     0, 1, 255, 255, 2, 3, 4,
                     255, 255, 255, 5, 255, 6, 7,
                     8, 9, 10, 11, 12, 13, 14,
                     15, 255, 255, 16, 17, 18]
train_cities = ['aachen', 'bremen', 'darmstadt', 'erfurt', 'hanover',
                'krefeld', 'strasbourg', 'tubingen', 'weimar', 'bochum',
                'cologne', 'dusseldorf', 'hamburg', 'jena', 'monchengladbach',
                'stuttgart', 'ulm', 'zurich']


# Copied and simplified from torch/vision/references/segmentation to compute mean IoU
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # For pseudo labels(which has 255), just don't let your network predict 255
            k = (a >= 0) & (a < n) & (b != 255)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu


def compute_MIX_map(output1, output2, Increment):
    MID = output2 - output1

    chance = MID[0:, 0:, 1:]
    chance = np.squeeze(chance, 2)

    backgroud1 = output1[0:, 0:, 0]
    backgroud1 = np.where(backgroud1 == 0, 0, 1)
    backgroud2 = output2[0:, 0:, 0]
    backgroud2 = np.where(backgroud2 == 0, 0, 1)
    backgroud = backgroud1 - backgroud2
    bk1 = np.where(backgroud == 1, 1, 0)
    bbk1 = np.where(backgroud == 1, 0, 1)
    bk2 = np.where(backgroud == -1, 1, 0)
    bbk2 = np.where(backgroud == -1, 0, 1)

    chance1 = np.where(chance >= Increment, 1, 0)
    chance2 = np.where(chance >= Increment, 0, 1)
    chance1 = np.expand_dims(chance1, 2).repeat(2, axis=2)
    chance2 = np.expand_dims(chance2, 2).repeat(2, axis=2)
    bk1 = np.expand_dims(bk1, 2).repeat(2, axis=2)
    bk2 = np.expand_dims(bk2, 2).repeat(2, axis=2)
    bbk1 = np.expand_dims(bbk1, 2).repeat(2, axis=2)
    bbk2 = np.expand_dims(bbk2, 2).repeat(2, axis=2)

    Final = np.multiply(output1, chance2) + np.multiply(output2, chance1)
    Final = np.multiply(output1, bk1) + np.multiply(Final, bbk1)
    Final = np.multiply(output2, bk2) + np.multiply(Final, bbk2)

    return Final


# Draw images/labels from tensors
def show(images, is_label, colors, std, mean):
    np_images = images.numpy()
    if is_label:
        # Map to RGB((N, d1, d2) = {0~20, 255} => (N, d1, d2, 3) = {0.0~1.0})
        # As for how I managed this, I literally have no clue,
        # but it seems to be working
        np_images = np_images.reshape((np_images.shape[0], np_images.shape[1], np_images.shape[2], 1))
        np_images = np.tile(np_images, (1, 1, 1, 3))
        np_images[np_images == 255] = len(colors) - 1  # Ignore 255
        np_images = np.array(colors)[np_images[:, :, :, 0]]
        np_images = np_images / 255.0
    else:
        # Denormalize and map from (N, 3, d1, d2) to (N, d1, d2, 3)
        np_images = np.transpose(np_images, (0, 2, 3, 1))
        np_images = np_images * std + mean
        if mean[0] > 1:
            np_images /= 255.0

    plt.imshow(np_images.reshape((np_images.shape[0] * np_images.shape[1], np_images.shape[2], np_images.shape[3])))
    plt.show()


# Save model checkpoints(supports amp)
def save_checkpoint(net, optimizer, lr_scheduler, is_mixed_precision, filename='temp.pt'):
    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'amp': amp.state_dict() if is_mixed_precision else None
    }
    torch.save(checkpoint, filename)


# Load model checkpoints(supports amp)
def load_checkpoint(net, optimizer, lr_scheduler, is_mixed_precision, filename):
    checkpoint = torch.load(filename, map_location="cuda:1")
    net.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if is_mixed_precision and checkpoint['amp'] is not None:
        amp.load_state_dict(checkpoint['amp'])


def generate_pseudo_labels(net, device, loader, old_net, num_classes, input_size,Increment, cbst_thresholds=None):
    # Generate pseudo labels and save to disk (negligible time compared with training)
    # Not very sure if there are any cache inconsistency issues (technically this should be fine)
    net.eval()

    # 1 forward pass (hard labels)
    if cbst_thresholds is None:  # Default
        cbst_thresholds = torch.tensor([0.99 for _ in range(num_classes)])
    cbst_thresholds = cbst_thresholds.to(device)
    net.eval()
    labeled_counts = 0
    ignored_counts = 0
    with torch.no_grad():
        for images, file_name_lists, heights, widths in tqdm(loader):
            images = images.to(device)
            outputs1 = net(images)['out']
            outputs2 = old_net(images)['out']

            predict1 = torch.nn.functional.interpolate(outputs1, size=input_size, mode='bilinear', align_corners=True)
            predict2 = torch.nn.functional.interpolate(outputs2, size=input_size, mode='bilinear', align_corners=True)



            for i in range(0, len(file_name_lists)):

                outputs1 = crop(predict1[i], 0, 0, heights[i], widths[i])  # Back to the original size
                outputs2 = crop(predict2[i], 0, 0, heights[i], widths[i])

                prediction1 = outputs1.softmax(dim=0)  # ! softmax
                prediction2 = outputs2.softmax(dim=0)

                temp1 = prediction1.max(dim=0)
                temp2 = prediction2.max(dim=0)
                pseudo_label1 = temp1.indices
                pseudo_label2 = temp2.indices

                values1 = temp1.values
                values2 = temp2.values
                for j in range(num_classes):
                    pseudo_label1[((pseudo_label1 == j) * (values1 < cbst_thresholds[j]))] = 255
                    pseudo_label2[((pseudo_label2 == j) * (values2 < cbst_thresholds[j]))] = 255

                # N x d1 x d2 x 2 (pseudo labels | original confidences)
                pseudo_label1 = pseudo_label1.unsqueeze(-1).float()
                pseudo_label1 = torch.cat([pseudo_label1, values1.unsqueeze(-1)], dim=-1)
                pseudo_label2 = pseudo_label2.unsqueeze(-1).float()
                pseudo_label2 = torch.cat([pseudo_label2, values2.unsqueeze(-1)], dim=-1)

                # Counting & Saving
                labeled_counts += (pseudo_label1[:, :, 0] != 255).sum().item()
                ignored_counts += (pseudo_label1[:, :, 0] == 255).sum().item()
                pseudo_label1 = pseudo_label1.cpu().numpy()
                pseudo_label2 = pseudo_label2.cpu().numpy()

                #Pseudo label enhancement
                pseudo_label = compute_MIX_map(pseudo_label1, pseudo_label2, Increment=Increment)

                np.save(file_name_lists[i], pseudo_label)

        # Return overall labeled ratio
        return labeled_counts / (labeled_counts + ignored_counts)
        # ratio for foreground


# Reimplemented (all converted to tensor ops) based on yzou2/CRST
def generate_class_balanced_pseudo_labels(net, device, loader, old_net, label_ratio, num_classes, input_size, Increment,
                                          down_sample_rate=16, buffer_size=100):
    # Max memory usage surge ratio has an upper limit of 2x (caused by array concatenation).
    # Keep a fixed GPU buffer size to achieve a good enough speed-memory trade-off,
    # since casting to cpu is very slow.
    # Note that tensor.expand() does not allocate new memory,
    # and that Python's list consumes at least 3 times the memory that a typical array would've required,
    # though it is 3 times faster in concatenations, it is rather slow in sorting,
    # thus the overall time consumption is similar.
    # buffer_size: GPU buffer size, MB.
    # down_sample_rate: Pixel sample ratio, i.e. pick one pixel every #down_sample_rate pixels.
    net.eval()
    buffer_size = buffer_size * 1024 * 1024 / 12  # MB -> how many pixels

    # 1 forward pass (sample predicted probabilities,
    # sorting here is unnecessary since there is relatively negligible time-consumption to consider)
    pseudo_label = torch.tensor([], dtype=torch.int64, device=device)
    pseudo_probability = torch.tensor([], dtype=torch.float32, device=device)
    probabilities = [np.array([], dtype=np.float32) for _ in range(num_classes)]
    with torch.no_grad():
        for images, _, heights, widths in tqdm(loader):
            images = images.to(device)
            outputs = net(images)['out']
            outputs = torch.nn.functional.interpolate(outputs, size=input_size, mode='bilinear', align_corners=True)

            # Generate pseudo labels (d1 x d2) and reassemble
            for i in range(0, len(heights)):
                prediction = crop(outputs[i], 0, 0, heights[i], widths[i])  # Back to the original size
                temp = prediction.softmax(dim=0)  # ! softmax
                temp = temp.max(dim=0)
                pseudo_label = torch.cat([pseudo_label, temp.indices.flatten()[:: down_sample_rate]])
                pseudo_probability = torch.cat([pseudo_probability, temp.values.flatten()[:: down_sample_rate]])

            # Count and reallocate
            if pseudo_probability.shape[0] > buffer_size:
                for j in range(num_classes):
                    probabilities[j] = np.concatenate((probabilities[j],
                                                       pseudo_probability[pseudo_label == j].cpu().numpy()))
                pseudo_label = torch.tensor([], dtype=torch.int64, device=device)
                pseudo_probability = torch.tensor([], dtype=torch.float32, device=device)

        # Final count
        for j in range(num_classes):
            probabilities[j] = np.concatenate((probabilities[j],
                                               pseudo_probability[pseudo_label == j].cpu().numpy()))

    # Sort (n * log(n) << n * label_ratio, so just sort is good) and find kc
    print('Sorting...')
    kc = []
    for j in range(num_classes):
        if len(probabilities[j]) == 0:
            with open('exceptions.txt', 'a') as f:
                f.write(str(time.asctime()) + '--' + str(j) + '\n')

    for j in tqdm(range(num_classes)):
        probabilities[j].sort()
        if label_ratio >= 1:
            kc.append(probabilities[j][0])
        else:
            if len(probabilities[j]) * label_ratio < 1:
                kc.append(0.00001)
            else:
                kc.append(probabilities[j][-int(len(probabilities[j]) * label_ratio) - 1])
    del probabilities  # Better be safe than...

    print(kc)
    return generate_pseudo_labels(net=net, device=device, old_net=old_net, loader=loader,
                                  cbst_thresholds=torch.tensor(kc),
                                  input_size=input_size, num_classes=num_classes, Increment=Increment)
