import numpy as np
import json
import os
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms




class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = np.array(k)


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'test.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)

    partitions = {'test': [], 'train': [], 'real_test': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])

    return partitions, labels


def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def project(q, r):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = Camera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y


class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='/datasets/speed_debug'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train', img_name=None):

        """ Loading image as PIL image. """
        if img_name is None:
            img_name = self.partitions[split][i]
        img_name = os.path.join(self.root_dir, 'images', split, img_name)
        image = Image.open(img_name).convert('RGB')
        return image

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train',qr= None, img_name=None, ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()


        # no pose label for test
        if partition == 'train':
            img = self.get_image(i)
            ax.imshow(img)
            q, r = self.get_pose(i)
        else:
            img = self.get_image(split="test", img_name=img_name )
            ax.imshow(img)
            if qr is None:
                return    
            q, r = qr
        xa, ya = project(q, r)
        ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
        ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
        ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

        return



class PyTorchSatellitePoseEstimationDataset(Dataset):

    """ SPEED dataset that can be used with DataLoader for PyTorch training. """

    def __init__(self, split='train', speed_root='', transform=None):

        if not has_pytorch:
            raise ImportError('Pytorch was not imported successfully!')

        if split not in {'train', 'test', 'real_test'}:
            raise ValueError('Invalid split, has to be either \'train\', \'test\' or \'real_test\'')

        with open(os.path.join(speed_root, split + '.json'), 'r') as f:
            label_list = json.load(f)

        self.sample_ids = [label['filename'] for label in label_list]
        self.train = split == 'train'

        if self.train:
            self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
                           for label in label_list}
        self.image_root = os.path.join(speed_root, 'images', split)

        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_name = os.path.join(self.image_root, sample_id)

        # note: despite grayscale images, we are converting to 3 channels here,
        # since most pre-trained networks expect 3 channel input
        pil_image = Image.open(img_name).convert('RGB')

        if self.train:
            q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
            y = np.concatenate([q, r])
        else:
            y = sample_id

        if self.transform is not None:
            torch_image = self.transform(pil_image)
        else:
            torch_image = pil_image

        return torch_image, y



