import os

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


class ImagePatcher:
    def __init__(self, input_dir, img_size):
        self.input_dir = input_dir
        self.img_size = img_size

    def batch_generator(self):
        all_images_batch = []
        all_transform_batch = []
        lens_image_batch = []
        crss = []
        output_files_names = []

        for filename in tqdm(os.listdir(self.input_dir)):
            output_file_name = filename.split("/")[-1].split(".")[0]
            input_file = os.path.join(self.input_dir, filename)

            with rasterio.open(input_file) as data:
                profile = data.profile
                crs = data.crs

                image_batch = []
                transform_batch = []

                for i in range(profile['width'] // self.img_size):
                    for j in range(profile['height'] // self.img_size):
                        win = Window(self.img_size * i, self.img_size * j, self.img_size, self.img_size)

                        img = data.read(window=win, out_dtype=np.uint8)

                        if img.ndim == 4:
                            img[img[..., 3] == 0] = 0
                        if img.shape[0] == 4:
                            img = img[:3, ...]

                        trn = data.window_transform(win)
                        image_batch.append(img)
                        transform_batch.append(trn)

            all_images_batch.append(image_batch)
            all_transform_batch.append(transform_batch)
            lens_image_batch.append(len(image_batch))
            crss.append(crs)
            output_files_names.append(output_file_name)

        self.lens_image_batch = lens_image_batch

        yield all_images_batch, all_transform_batch, crss, output_files_names
