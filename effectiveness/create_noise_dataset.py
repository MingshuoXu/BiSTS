import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]


import cv2
from skimage.util import random_noise


BENCHMARK_FOLDER = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                  'lgmd_benchmark'))
NOISE_VIDEOS_FOLDER = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 'BiSTS_extend',
                                                   'effe_noise_videos'))

RAW_DATA_FILE_NAME = os.path.join(os.path.dirname(__file__), file_name_without_ext+'.json')

NOISE_CLASSES = ['gaussian', 'salt-and-pepper']
NOISE_INTENSITIES = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3]

MODEL_LIST = ['BasicLGMD', 'pLGMD', 'LCR_LGMD', 'LCR_LGMD_N']
RAW_INPUT_VIDEOS = [
    'looming ball against static bg\\ha-dark1.mp4',
    'basic test\\looming square\\dark_looming.mp4',
    'vehicle against driving sense\\tc08.mp4'
]


def create_noise_datasets():
    def _add_noise_and_save(vidPth, noiseType=None, noiseIntensity=None):
        # in
        objVid = cv2.VideoCapture(vidPth)
        width = int(objVid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(objVid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # out
        outPth = os.path.join(NOISE_VIDEOS_FOLDER, f'{os.path.basename(vidPth)[:-4]}',
                              f'{os.path.basename(vidPth)[:-4]}_{noiseType}_{noiseIntensity}.mp4')
        os.makedirs(os.path.dirname(outPth), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outVid = cv2.VideoWriter(outPth, fourcc, 30.0, (width, height))

        # add noise
        while objVid.isOpened():
            ret, colorImg = objVid.read()
            if not ret:
                break

            colorImg = colorImg.astype(float) / 255.0  # Normalize to [0, 1]
            if noiseType == 'gaussian':
                img_with_noise = random_noise(colorImg, mode='gaussian', var=noiseIntensity)
            elif noiseType == 'salt-and-pepper':
                img_with_noise = random_noise(colorImg, mode='s&p', amount=noiseIntensity)
            img_with_noise *= 255.0

            outVid.write(img_with_noise.astype('uint8'))

    for vid in RAW_INPUT_VIDEOS:
        vidPth = os.path.join(BENCHMARK_FOLDER, vid)
        for noise in NOISE_CLASSES:
            for intensy in NOISE_INTENSITIES:
                _add_noise_and_save(vidPth, noise, intensy)


if __name__ == "__main__":
    create_noise_datasets()
    
