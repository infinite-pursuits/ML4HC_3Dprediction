from torch.utils.data import Dataset
import nibabel
import numpy as np
import random
from PIL import Image
import torch
from skimage import exposure, filters, io
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrainImages(Dataset):
    def __init__(self, image_dir, label_dict, subjects, prep=False, augment=False, train_data=False, flipping=False,
                 rotation=False, translation=False, gaussian=False, mean_image=False):

        self.image_dir = image_dir
        self.flipping = flipping
        self.rotation = rotation
        self.translation = translation
        self.train_data = train_data
        self.subjects = subjects
        self.label_dict = label_dict
        self.prep = prep
        self.augment = augment
        self.gaussian = gaussian
        self.sub_mean = mean_image
    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image_path = self.image_dir[idx]
        subject = self.subjects[idx]
        target_label = self.label_dict[subject]
        img_orig = nibabel.freesurfer.mghformat.MGHImage.from_filename(image_path)
        image = img_orig.get_data().astype(np.float64)
        im = Image.fromarray(image[0])

        if self.augment:
            flip = random.random() > 0.5
            angle = random.uniform(-10, 10)
            dx = np.round(random.uniform(-25, 25))
            dy = np.round(random.uniform(-25, 25))

            if self.train_data:
                if self.flipping and flip:
                    im = im.transpose(0)
                if self.rotation:
                    im = im.rotate(angle)
                if self.translation:
                    im = im.transform((256, 256), 0, (1, 0, dx, 0, 1, dy))
        if self.prep:
            im = np.array(im, np.float64, copy=False)
            if self.sub_mean:
                mean_image = io.imread('avg_image.png')[:,:,0]
                im = im - mean_image
            min_im = np.min(im)
            max_im = np.max(im)
            im = (im - min_im) / (max_im - min_im + 1e-4)

        if self.gaussian:
            gaussian_flag = random.random() > 0.5
            if self.train_data and gaussian_flag:
                sigma_rand = random.uniform(0.65, 1.0)
                im_sigma = filters.gaussian(im, sigma=sigma_rand)
                gamma_rand = random.uniform(1.6, 2.4)
                im_sigma_gamma = exposure.adjust_gamma(im_sigma, gamma_rand)
                im = (im_sigma_gamma - np.min(im_sigma_gamma)) / (
                            np.max(im_sigma_gamma) - np.min(im_sigma_gamma) + 1e-4)

        imgz = torch.from_numpy(np.array(im, np.float64, copy=False).reshape((1, 256, 256))).float()
        target_label = np.array(list(target_label))
        target_label = torch.from_numpy(target_label).float()
        sample = {'x': imgz, 'y': target_label}
        return sample
