import glob

from PIL import Image
from torch.utils.data import Sampler, Dataset
from transforms import *
from torchvision.transforms import transforms
from utils import show_images

class ZSSRDataset(Dataset):
    def __init__(self, source_image, sr_factor):
        super(ZSSRDataset, self).__init__()
        self.sr_factor = sr_factor
        self.source_image = source_image
        smaller_side = min(self.source_image.size[0: 2])
        larger_side = max(self.source_image.size[0: 2])

        factors = []
        for i in range(smaller_side // 5, smaller_side + 1):
            downsampled_smaller_side = i
            zoom = float(downsampled_smaller_side) / smaller_side
            downsampled_larger_side = round(larger_side * zoom)
            if downsampled_smaller_side % self.sr_factor == 0 and \
                    downsampled_larger_side % self.sr_factor == 0:
                factors.append(zoom)

        hr_lr_pairs = []
        for zoom in factors:
            hr = self.source_image.resize((int(self.source_image.size[0] * zoom),
                                  int(self.source_image.size[1] * zoom)),
                                 resample=Image.BICUBIC)

            lr = hr.resize((int(hr.size[0] / self.sr_factor),
                            int(hr.size[1] / self.sr_factor)),
                           resample=Image.BICUBIC)

            hr_lr_pairs.append((hr, lr))

        self.hr_lr_pairs = hr_lr_pairs
        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip()])

    def __getitem__(self, index):
        return self.transform(self.hr_lr_pairs[index])

    def __len__(self):
        return len(self.hr_lr_pairs)

    @classmethod
    def from_image(cls, img, sr_factor):
        pil = transforms.ToPILImage()(transforms.ToTensor()(img))
        return ZSSRDataset(pil, sr_factor)

    def show_pairs(self):
        number_of_pairs = len(self)
        indexes = random.sample(range(number_of_pairs), k=4)
        pairs = [self.__getitem__(index) for index in indexes]
        lrs = [lr for _, lr in pairs]
        hrs = [hr for hr, _ in pairs]
        show_images(lrs + hrs, 2, 4)

    def concat(self, dataset):
        self.hr_lr_pairs += dataset.hr_lr_pairs
        return self

class ZSSRSampler(Sampler):
    def __init__(self, dataset):
        super(ZSSRSampler, self).__init__(dataset)
        self.dataset = dataset
        sizes = np.float32([(hr.size[0] * hr.size[1] / float(
            self.dataset.source_image.size[0] * self.dataset.source_image.size[1])) for hr, lr in self.dataset.hr_lr_pairs])
        self.pair_probabilities = sizes / np.sum(sizes)

    def __iter__(self):
        while True:
            yield random.choices(self.dataset, weights=self.pair_probabilities, k=1)[0]

def downsample_all_images(heigh_res_img_path, kernel, sr_factor, output_folder, adding_noise=False):
    downsampling = {
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "box": Image.BOX,
        "cubic": Image.CUBIC,
        "lanczos": Image.LANCZOS
    }
    available_downsampling = ["bilinear", "nearest", "box", "lanczos"]
    images = glob.glob(f"{heigh_res_img_path}/img_*.png")

    with open(f"{heigh_res_img_path}/{output_folder}/kernels.txt","w") as kernels_file:
        for hr_img_path in images:
            if kernel == "random":
                kernel_name = available_downsampling[random.randint(0, len(available_downsampling) - 1)]
                downsampling_kernel = downsampling[kernel_name]
            else:
                kernel_name = kernel
                downsampling_kernel = downsampling[kernel]

            hr_img = Image.open(hr_img_path)
            lr_img = hr_img.resize((int(hr_img.size[0] / sr_factor),
                                    int(hr_img.size[1] / sr_factor)),
                                   resample=downsampling_kernel)

            lr_img.save(f"{heigh_res_img_path}/{output_folder}/{hr_img_path[-11:]}")
            kernels_file.write(f"{hr_img_path} - {kernel_name}")


class VDSRDataset(Dataset):
    def __init__(self, path_to_hr_images, path_to_lr_images, transforms):
        super(VDSRDataset, self).__init__()
        self.hr_images = []
        self.lr_images = []
        self.transforms = transforms
        hr_images = sorted(glob.glob(f"{path_to_hr_images}/img_*.png"))
        lr_images = sorted(glob.glob(f"{path_to_lr_images}/img_*.png"))

        for lr_image_path in lr_images:
            lr = Image.open(lr_image_path)
            lr = lr.resize([lr.size[0] * 2, lr.size[1] * 2],
                           resample=Image.BICUBIC)
            self.lr_images.append(lr)

        for hr_image_path in hr_images:
            hr = Image.open(hr_image_path)
            self.hr_images.append(hr)

    def __getitem__(self, index):
        lr, hr = self.transforms(self.lr_images[index]), self.transforms(self.hr_images[index])
        if lr.shape[1] > lr.shape[2]:
            lr = lr.permute(0, 2, 1)

        if hr.shape[1] > hr.shape[2]:
            hr = hr.permute(0, 2, 1)

        assert lr.shape[1] == hr.shape[1] and lr.shape[2] == hr.shape[2]
        return lr, hr

    def __len__(self):
        return len(self.hr_images)

if __name__ == "__main__":
    downsample_all_images("./data/BSD100", kernel="random", sr_factor=2, output_folder="unknown_kernel_sr2")
    downsample_all_images("./data/BSD100", kernel="random", sr_factor=4, output_folder="unknown_kernel_sr4")