# -----------------------------------------------------------------------------
# File          : crop_data.py
# Description   : crops data into 320x320 images around potential plumes
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
from util import *
from pathlib import Path
from tqdm.contrib.concurrent import thread_map
import numpy as np
from datetime import datetime
import cv2


class Data:
    def __init__(self, data_file):
        self.file_name = Path(data_file).stem
        
        with np.load(data_file) as data:
            self.scanline = data['dims'][0]
            self.ground_pixel = data['dims'][1]

            self.lat, self.lon = data['lat'], data['lon']
            self.so2, self.so2_flag = data['so2'], data['so2_flag']
            self.qa_value = data['qa_value']
            self.sza, self.vza = data['sza'], data['vza']

        trim = 25
        self.lat = self.lat[:, trim:-trim]
        self.lon = self.lon[:, trim:-trim]
        self.so2 = self.so2[:, trim:-trim]
        self.so2_flag = self.so2_flag[:, trim:-trim]
        self.qa_value = self.qa_value[:, trim:-trim]
        self.sza = self.sza[:, trim:-trim]
        self.vza = self.vza[:, trim:-trim]
        self.ground_pixel -= trim * 2

        self.so2 = np.ma.masked_where(self.sza > 70, self.so2)
        self.so2_flag = np.ma.filled(self.so2_flag, 0)
        self.so2_flag[self.so2_flag > 1] = 1

        self.centerlon = self.lon[int(self.scanline/2), int(self.ground_pixel/2)]

    def calc_blobs(self, blursize1=11, blursize2=5, threshold=1, minblobsize=100):
        # image = self.so2_flag.astype(np.float32)
        image = np.ma.filled(self.so2, 0)
        image = cv2.blur(image, (blursize1, blursize1))
        ret, image = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
        image = cv2.blur(image, (blursize2, blursize2))
        ret, image = cv2.threshold(image, 0.1, 1, cv2.THRESH_BINARY)

        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(image.astype(np.uint8), connectivity=8)
        labels_im = np.ma.masked_where(labels_im == 0, labels_im)
        blob_labels = []
        self.blob_centroids = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < minblobsize:
                labels_im = np.ma.masked_where(labels_im == i, labels_im)
            else:
                blob_labels.append(i)
                self.blob_centroids.append(centroids[i])
        for new_label, label in enumerate(blob_labels):
            labels_im[labels_im == label] = new_label + 1

        self.blobs_im = labels_im
        self.num_blobs = len(blob_labels)

    def crop_blobs(self, cropsize=320):
        blob_crops = []
        for label in range(self.num_blobs):
            coords = np.column_stack(np.where(self.blobs_im == label + 1))

            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)

            radius = cropsize // 2
            x_center = np.clip((x_min + x_max)//2, radius, self.scanline - radius)
            y_center = np.clip((y_min + y_max)//2, radius, self.ground_pixel - radius)
            blob_crops.append([x_center - radius, x_center + radius, y_center - radius, y_center + radius])
        return blob_crops

    def save(self, crop, p_save_dir, p_save_name):
        # crop data
        lon = self.lon[crop[0]:crop[1], crop[2]:crop[3]]
        lat = self.lat[crop[0]:crop[1], crop[2]:crop[3]]
        so2 = self.so2[crop[0]:crop[1], crop[2]:crop[3]]
        so2_flag = self.so2_flag[crop[0]:crop[1], crop[2]:crop[3]]
        qa_value = self.qa_value[crop[0]:crop[1], crop[2]:crop[3]]
        sza = self.sza[crop[0]:crop[1], crop[2]:crop[3]]
        vza = self.sza[crop[0]:crop[1], crop[2]:crop[3]]

        so2_flag = so2_flag.astype(np.ubyte)

        save_path = f'{p_save_dir}/{p_save_name}'
        # print(f'saving to {save_path}')
        np.savez(
            save_path,
            dims=np.array([crop[1]-crop[0], crop[3]-crop[2]]),
            lat=lat, lon=lon, so2=so2, so2_flag=so2_flag,
            qa_value=qa_value, sza=sza, vza=vza
        )


def cropfile(f):
    data = Data(f)
    data.calc_blobs(blursize1=7, blursize2=3, threshold=0.3, minblobsize=400)
    for crop in data.crop_blobs():
        save_name = f'{f.stem}_{crop[1]-crop[0]}x{crop[3]-crop[2]}_{crop[0]:04}_{crop[2]:02}'
        data.save(crop, CROPPED_PATH, save_name)
    return 0


def main():
    years = [year for year in Path(DATA_PATH).iterdir() if year.name != 'temp']
    months = [month for year in years for month in year.iterdir()]
    days = [day for month in months for day in month.iterdir()]
    files = [f for day in days for f in day.iterdir() if f.suffix == '.npz']
   
    print('cropping...')
    _ = thread_map(cropfile, files)
    print('done')


if __name__ == '__main__':
    main()
