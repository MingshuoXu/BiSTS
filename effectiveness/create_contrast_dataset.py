import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]

from psychopy import visual, core
import cv2
from tqdm import tqdm
import numpy as np
import concurrent.futures


BG_FOLDER = os.path.join(os.path.dirname(itemPth), 
                         '7_Dataset', 'data-from-LGMD_P_ON_OFF')
NEW_DATASET_FOLDER = os.path.join(os.path.dirname(itemPth), 
                                  '7_Dataset', 'lgmd_benchmark', 'looming square against nature scenarios')
BENCHMARK_FOLDER = os.path.join(os.path.dirname(itemPth), 
                                  '7_Dataset', 'lgmd_benchmark')
os.makedirs(NEW_DATASET_FOLDER, exist_ok=True)

RAW_DATA_FILE_NAME = os.path.join(os.path.dirname(__file__), file_name_without_ext+'.json')


class SpinningDrum:
    def __init__(self, win, image_path, size=(927.0, 251.0), pos=(0,0), 
                 speed=2, direction='left', smoothing=True):
        """
        参数:
        - win: PsychoPy 窗口
        - image_path: 纹理图片路径
        - size: 图像尺寸
        - pos: 初始位置
        - speed: 滚动速度
        - direction: 滚动方向 ('left', 'right')
        """
        self.win = win
        self.size = size
        self.base_pos = pos
        self.speed = speed
        self.direction = -1 if direction == 'left' else 1
        
        # 平滑设置
        smooth_filter = 'linear' if smoothing else 'nearest'

        # 创建两个图像实例实现无缝循环
        self.drum1 = visual.ImageStim(
            win=win,
            image=image_path,
            size=size,
            pos=pos,
            opacity=1.0,
            interpolate=smoothing,      # 启用插值
            # filter=smooth_filter,       # 设置过滤质量
            units='pix'                 # 确保像素级精度
        )
        
        self.drum2 = visual.ImageStim(
            win=win,
            image=image_path,
            size=size,
            pos=(pos[0] + size[0], pos[1]),
            opacity=1.0,
            interpolate=smoothing,      # 启用插值
            # filter=smooth_filter,       # 设置过滤质量
            units='pix'                 # 确保像素级精度
        )
        
        self.current_position = 0
    
    def set_speed(self, new_speed):
        """设置滚动速度"""
        self.speed = new_speed
    
    def set_direction(self, direction):
        """设置滚动方向"""
        self.direction = -1 if direction == 'left' else 1
    
    def update(self):
        """更新鼓的位置（每帧调用）"""
        self.current_position += self.direction * self.speed
        
        # 循环逻辑
        if abs(self.current_position) >= self.size[0]:
            self.current_position = 0
        
        # 更新两个图像的位置
        x_pos = self.current_position
        self.drum1.pos = (self.base_pos[0] + x_pos, self.base_pos[1])
        self.drum2.pos = (self.base_pos[0] + x_pos + self.size[0], self.base_pos[1])
    
    def draw(self):
        """绘制鼓"""
        self.drum1.draw()
        self.drum2.draw()
    
    def reset(self):
        """重置位置"""
        self.current_position = 0
        self.drum1.pos = self.base_pos
        self.drum2.pos = (self.base_pos[0] + self.size[0], self.base_pos[1])


class CreateNatualDataset:
    def __init__(self, drum_texture_path, target_int_gray: int=0):
        self.texture_path = drum_texture_path

        # Set up display parameters
        self.fps = 30.0
        self.win_size = (432, 240)  # Adjust to your screen resolution

        
        # Create the target (using Rect for rectangular target)
        self.init_target_size = 16  # initial width
        self.target_int_gray = target_int_gray  # integer gray level (0-255)
        self.target_float_gray = float(target_int_gray)/255*2-1  



        # Animation parameters
        self.V_B = 20  # background rotation speed (degrees/sec)
        self.Scale = 927 / 360  # scale factor for drum rotation


        self.win = visual.Window(
            size=self.win_size,
            color=(-1, -1, -1),
            units='pix',
            fullscr=False,
            screen=0,
            waitBlanking=False,
            allowGUI=False,
        )


        self.drum = SpinningDrum(
            win=self.win,
            image_path=drum_texture_path,
            size=(927.0, 251.0),  # Original size from your code
            pos=(0, 0), 
            speed= self.V_B * self.Scale / self.fps, 
            direction='left'
        )

        self.target = visual.Rect(
            win=self.win,
            width=self.init_target_size,
            height=self.init_target_size,
            fillColor=(self.target_float_gray, self.target_float_gray, self.target_float_gray),  # White target (PsychoPy uses -1 to 1 range)
            lineColor=None,
            pos=(0, 0)  # Initial position will be updated
        )


        # Create a clock to track time
        self.clock = core.Clock()

    def update_target_size(self, time_idx, focal=90, max_len=225, total_motion_frames=43):
        ''' 计算前景位置 '''
        max_depth = focal * max_len / self.init_target_size  # 物体初始距离
        speed = (max_depth - focal) / total_motion_frames  # 迫近速度

        if time_idx < 20:
            foreSize = self.init_target_size
        elif time_idx <= 63:
            dis = max_depth - speed * (time_idx-20) # 当前距离
            foreSize = int(focal * max_len / dis) # 真实的边长
        else:
            foreSize = max_len

        return foreSize

    # Main animation loop
    def run_animation(self, duration=3.0):
        """Run the animation for specified duration (seconds)"""
        
        # Reset the clock
        self.clock.reset()
        
        while self.clock.getTime() < duration:
            # Get current time
            t = self.clock.getTime() * self.fps
            
            # Update drum orientation
            # self.drum.pos = (self.update_drum_orientation(t), 0)
            
            # Update target size
            target_size = self.update_target_size(t, self.win_size[0], self.win_size[1])
            self.target.width = target_size
            self.target.height = target_size
            
            # Draw stimuli (order matters for layering)
            self.drum.update()  # Update drum position
            self.drum.draw()  # Draw drum first (background)
            self.target.draw()  # Draw target on top
            
            # Flip the buffer to display
            self.win.flip()
            
            # Check for quit event
            # if len(self.win.keys) > 0:
            #     if 'escape' in self.win.keys:
            #         break

    # Alternative version with movie recording
    def run_animation_with_recording(self, duration=3.0):
        """Run animation and save as movie file"""

        save_name = os.path.join(NEW_DATASET_FOLDER,
            f'natural-{os.path.splitext(os.path.basename(self.texture_path))[0]}-fgr_gray={self.target_int_gray}.mp4')
        # 设置视频编写器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_name, fourcc, self.fps,
                              (int(self.win.size[0]), int(self.win.size[1]))
                              )

        for t in range(int(duration * self.fps)):
            
            # Update target size
            target_size = self.update_target_size(t)
            self.target.width = target_size
            self.target.height = target_size
            
            # Draw stimuli (order matters for layering)
            self.drum.update()  # Update drum position
            self.drum.draw()  # Draw drum first (background)
            self.target.draw()  # Draw target on top

            # 捕获当前帧
            frame_data = self.win._getFrame(buffer='back')
            frame_array = np.array(frame_data)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()


    def __del__(self):
        self.win.close()
        self.drum = None
        self.target = None



def _create_dataset_for_one_contrast(pth, fore_int_gray):
    obj = CreateNatualDataset(pth, fore_int_gray)
    obj.run_animation_with_recording(duration=3.0)
    del(obj)

    return ...


def create_natual_dataset():
    foreground_contrasts = [0, 255]

    bgPthSets = {}
    for root, dirs, files in os.walk(BG_FOLDER):
        for file in files:
            if file.endswith('.jpg'):
                bgPthSets.update({file: os.path.join(root, file)})

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for fore_gray in foreground_contrasts:
            for pth in bgPthSets.values():
                future = executor.submit(_create_dataset_for_one_contrast, pth, fore_gray)
                futures.append(future)


        for future in concurrent.futures.as_completed(futures):
            future.result()
            

def create_contrast_looming():
    vid_reader = cv2.VideoCapture(os.path.join(BENCHMARK_FOLDER, 'basic test', 'looming square', 'dark_looming.mp4'))
    vid_H = int(vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_W = int(vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = []
    while vid_reader.isOpened():
        ret, frame = vid_reader.read()
        if not ret:
            break
        frames.append(frame)
    vid_reader.release()
      
    foreground_masks = [frame < 128 for frame in frames]

    for fore_contrast in (0, 32, 64, 96, 128, 160, 192, 224, 255):
        for back_contrast in (0, 32, 64, 96, 128, 160, 192, 224, 255):
            if fore_contrast == back_contrast:
                continue
            vid_name = os.path.join(os.path.dirname(itemPth), '7_Dataset', 'BiSTS_extend', 'contrast looming',
                                    f'looming_square_fgr={fore_contrast}_bgr={back_contrast}.mp4')
            vid_writer = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (vid_W, vid_H), isColor=True)
            for mask in foreground_masks:
                frame = np.full((vid_H, vid_W, 3), back_contrast, dtype=np.uint8)
                frame[mask] = fore_contrast
                vid_writer.write(frame)
            vid_writer.release()

    
if __name__ == "__main__":
    # obj = CreateNatualDataset(os.path.join(BG_FOLDER, "0040_120518-135330_tonemapped.jpg"))
    # obj.run_animation_with_recording(duration=3.0)

    create_natual_dataset()
    create_contrast_looming()

