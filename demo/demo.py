import os
import sys
ITEM_PTH = os.path.dirname(os.path.dirname(__file__)) # Project root directory
if ITEM_PTH not in sys.path:
    sys.path.append(ITEM_PTH) # Add the parent directory to sys.path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import torch # Deep learning framework
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

from utils import visualize_LGMD, get_input_by_GUI
from model import bists_lgmd, existing_lgmd



def main(model, vidPath=None):
    # Video path
    if vidPath is None:
        vidPath = get_input_by_GUI(initDir=os.path.join(os.path.dirname(ITEM_PTH), 
                                                    '7_Dataset', 'lgmd_benchmark'))
    
    # Load video
    objVid = cv2.VideoCapture(vidPath)
    
    # Create the visualizer
    visualizer = visualize_LGMD(model.__class__.__name__)

    paused = False
    running = True

    def on_key(event):
        nonlocal paused, running
        if event.key == ' ':
            paused = not paused
            state = 'paused' if paused else 'resumed'
            print(f'Playback {state}.')
        elif event.key in ('q', 'escape'):
            running = False
            print('Exit requested.')

    visualizer.fig.canvas.mpl_connect('key_press_event', on_key)

    # Run
    while objVid.isOpened() and running:
        if paused:
            plt.pause(0.1)
            continue

        ret, colorImg = objVid.read()
        if not ret:
            break

        gray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(DEVICE) # Convert to a tensor and add batch/channel dimensions

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()  
        time_start = time.perf_counter()
        
        k = model(gray_tensor)

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()  
        timeCost = time.perf_counter() - time_start

        if isinstance(k, torch.Tensor):
            k = k.cpu().numpy()  # Convert to a NumPy array
        visualizer.update(colorImg, k, timeCost, objVid.get(cv2.CAP_PROP_POS_FRAMES))
        # Simulate a real-time frame interval
        time.sleep(0.001)

    # Close the figure
    visualizer.close()
    objVid.release()



if __name__ == '__main__':
    # Instantiate the model
    model = bists_lgmd.BiSTS_LGMD()
    # model = existing_lgmd.BasicLGMD()
    # model = existing_lgmd.LGMD_P_ON_OFF()
    model.to(DEVICE)

    # vidPath = os.path.join(os.path.dirname(ITEM_PTH), 
                        #    '7_Dataset', 'over 400x400 synthetic stimuli', 'looming-s1-0-b255.avi')
    # main(model, vidPath)
    main(model)
