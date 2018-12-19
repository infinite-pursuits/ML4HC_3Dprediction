from torch.utils.data import Dataset
import nibabel
import numpy as np
import random
from PIL import Image
import torch
from skimage import exposure, filters
from skimage import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BrainImages(Dataset):
    def __init__(self, datadir, label_dict, subjects, slice_type='top_to_bottom', prep=False, augment=False, train_data=False,
                 flipping=False, rotation=False, translation=False, gaussian=False, sub_mean_img=False):

        self.flipping = flipping
        self.rotation = rotation
        self.translation = translation
        self.train_data = train_data
        self.subjects = subjects
        self.label_dict = label_dict
        self.prep = prep
        self.augment = augment
        self.gaussian = gaussian
        self.slice_type = slice_type
        self.sub_mean_img = sub_mean_img
        self.mean_img = io.imread('avg_image.png')[:, :, 0]
        self.datadir = datadir

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        image_path = self.datadir + '/'+self.slice_type + "/" + subject + "/T1_images"
        # print("img path {}".format(image_path))
        target_label = self.label_dict[int(subject)]
        # print("TL {}".format(target_label))
        img_tensor = torch.empty((110, 256, 256))
        # print("IMGtensor of size {} formed".format(img_tensor.shape))
        for i in range(100, 210):
            img_orig = nibabel.freesurfer.mghformat.MGHImage.from_filename(image_path + "/image{}.mgz".format(i))
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
                if self.sub_mean_img:
                    im = im - self.mean_img
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

            """print("idx {} target_label {}\n".format(idx, target_label))
            plt.figure()
            plt.imshow(im)
            plt.show()"""

            im = torch.from_numpy(np.array(im, np.float64, copy=False).reshape((256, 256))).float()
            img_tensor[i - 100, :, :] = im
        img_tensor = img_tensor.reshape((1, 110, 256, 256))
        img_tensor = img_tensor
        target_label = np.array(list(target_label))
        target_label = torch.from_numpy(target_label).float()
        sample = {'x': img_tensor, 'y': target_label}
        return sample
