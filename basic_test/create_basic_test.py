import os
import json

import cv2
import numpy as np



FILE_PTH = os.path.dirname(__file__)


TARGET_PTH = os.path.join(os.path.dirname(os.path.dirname(FILE_PTH)), 
                          "7_Dataset", "lgmd_benchmark", "basic test")


SUB_CLASSES = ['looming square', 'receding square', 'translating bar', 'linear deformation']

vid_height = 240
vid_width = 432
fps = 30
total_frame = 50

looming_square = ["dark_looming.mp4", "light_looming.mp4"]
receding_square = ["dark_reveding.mp4", "light_receding.mp4"]
translating_bar = ["light_translating_vertical.mp4", "dark_translating_vertical.mp4",
                    "light_translating_horizontal.mp4", "dark_translating_horizontal.mp4"]
linear_deformation = ["dark_elongating_vertical.mp4", "light_elongating_vertical.mp4", 
                      "dark_shortening_vertical.mp4", "light_shortening_vertical.mp4",
                      "dark_elongating_horizontal.mp4", "light_elongating_horizontal.mp4", 
                      "dark_shortening_horizontal.mp4", "light_shortening_horizontal.mp4"]


def ensure_output_dirs():
    os.makedirs(TARGET_PTH, exist_ok=True)
    for cls_name in SUB_CLASSES:
        os.makedirs(os.path.join(TARGET_PTH, cls_name), exist_ok=True)


def get_bg_value(filename):
    return 255 if filename.startswith("dark") else 0


def render_frames_from_masks(mask_frames, bg):
    fg = 0 if bg == 255 else 255
    frames = []
    for mask in mask_frames:
        frame = np.full((vid_height, vid_width), bg, dtype=np.uint8)
        frame[mask] = fg
        frames.append(frame)
    return frames


def calc_mean_change(frames):
    if len(frames) < 2:
        return 0.0
    changes = []
    for i in range(1, len(frames)):
        diff = float(np.mean(cv2.absdiff(frames[i], frames[i - 1])))
        if diff > 1e-2:
            changes.append(diff)
    if not changes:
        return 0.0

    return float(np.mean(changes))


def write_video(video_path, frames):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, float(fps), (vid_width, vid_height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    out.release()


def build_looming_masks():
    # 基于简单的透视投影模型生成looming square的mask序列
    # focal / init_depth  = proj / object_size
    
    focal = 1
    init_proj = 10
    init_depth = focal / init_proj * (vid_width + vid_height) / 2.0
    speed = (init_depth - focal*1.5) / (total_frame-1-5)

    masks = []

    for t in range(total_frame):
        if t >= total_frame-5:
            depth = focal*1.5
        else:
            depth = init_depth - speed * t
        proj = focal / depth * (vid_width + vid_height) / 2.0
        side = max(2, int(round(proj)))

        cx, cy = vid_width // 2, vid_height // 2
        x0 = max(0, cx - side // 2)
        x1 = min(vid_width, cx + side // 2)
        y0 = max(0, cy - side // 2)
        y1 = min(vid_height, cy + side // 2)

        mask = np.zeros((vid_height, vid_width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        masks.append(mask)

    return masks


def build_translating_masks(orientation, mean_change):
    masks = []

    # `orientation` means motion direction.
    if orientation == "vertical":
        # Move along y; keep cross-motion width fixed at half image width.
        # (velocity * vid_width * 255) / (vid_height * vid_width) = mean_change
        velocity = mean_change * vid_height / 255

        rect_h = velocity * 1.2

        moving_frame = int(vid_height / velocity + 1)
        moving_frame = max(1, min(total_frame, moving_frame))
        static_frame = total_frame - moving_frame

        masks = [np.zeros((vid_height, vid_width), dtype=bool) for _ in range(static_frame//2)]
        for t in range(moving_frame):

            y = int(velocity * t)
            y0 = max(0, y)
            y1 = min(vid_height, int(y + rect_h))
            mask = np.zeros((vid_height, vid_width), dtype=bool)
            if y1 > y0:
                mask[y0:y1, :] = True
            masks.append(mask)
        masks += [np.zeros((vid_height, vid_width), dtype=bool) for _ in range(static_frame - static_frame//2)]
    else:
        velocity = mean_change * vid_height / 255
        rect_w = velocity * 1.2

        moving_frame = int(vid_width / velocity + 1)
        moving_frame = max(1, min(total_frame, moving_frame))
        static_frame = total_frame - moving_frame

        masks = [np.zeros((vid_height, vid_width), dtype=bool) for _ in range(static_frame//2)]
        for t in range(moving_frame):

            x = int(velocity * t)
            x0 = max(0, x)
            x1 = min(vid_width, int(x + rect_w))
            mask = np.zeros((vid_height, vid_width), dtype=bool)
            if x1 > x0:
                mask[:, x0:x1] = True
            masks.append(mask)
        masks += [np.zeros((vid_height, vid_width), dtype=bool) for _ in range(static_frame - static_frame//2)]

    return masks


def build_linear_deformation_masks(mode, orientation, mean_change):
    masks = []

    step_v = mean_change * vid_height / 0.9 / 255 * 2
    step_h = mean_change * vid_width / 0.9 / 255 * 2
    step_v = max(step_v, 1e-6)
    step_h = max(step_h, 1e-6)

    if orientation == "vertical":
        # Deformation speed is determined only by image size and frame count.
        # (vid_width*0.9) * 255 * velocity / (vid_height * vid_width) = mean_change
        lengths = np.arange(0, vid_height + step_v, step_v)
        moving_frame = len(lengths)
        moving_frame = max(1, min(total_frame, moving_frame))
        lengths = lengths[:moving_frame]
        static_frame = total_frame - moving_frame
        if static_frame >= 2:
            lengths = np.concatenate([
                np.zeros(static_frame // 2),
                lengths,
                np.ones(static_frame - static_frame // 2) * vid_height,
            ])
        if mode == "shortening":
            lengths = lengths[::-1]

        # One-sided vertical deformation: keep top edge fixed, move bottom edge.
        for length_float in lengths:
            y1 = min(vid_height, max(1, int(round(length_float))))

            mask = np.zeros((vid_height, vid_width), dtype=bool)
            mask[:y1, int(0.05*vid_width):int(0.95*vid_width)] = True
            masks.append(mask)
    else:
        # Deformation speed is determined only by image size and frame count.
        lengths = np.arange(0, vid_width + step_h, step_h)
        moving_frame = len(lengths)
        moving_frame = max(1, min(total_frame, moving_frame))
        lengths = lengths[:moving_frame]
        static_frame = total_frame - moving_frame
        if static_frame >= 2:
            lengths = np.concatenate([
                np.zeros(static_frame // 2),
                lengths,
                np.ones(static_frame - static_frame // 2) * vid_width,
            ])
        if mode == "shortening":
            lengths = lengths[::-1]

        # One-sided horizontal deformation: keep left edge fixed, move right edge.
        for length_float in lengths:
            x1 = min(vid_width, max(1, int(round(length_float))))

            mask = np.zeros((vid_height, vid_width), dtype=bool)
            mask[int(0.05*vid_height):int(0.95*vid_height), :x1] = True
            masks.append(mask)

    return masks


def create_basic_test_dataset():
    ensure_output_dirs()

    looming_masks = build_looming_masks()

    # 1) looming square: use full contrast as baseline
    looming_changes = []
    for filename in looming_square:
        bg = get_bg_value(filename)
        frames = render_frames_from_masks(looming_masks, bg=bg)
        mean_change = calc_mean_change(frames)
        looming_changes.append(mean_change)

        video_path = os.path.join(TARGET_PTH, "looming square", filename)
        write_video(video_path, frames)

    target_change = float(np.mean(looming_changes)) * 2


    # 2) receding square: direct time-reverse of looming
    for filename in receding_square:
        bg = get_bg_value(filename)
        frames = render_frames_from_masks(looming_masks, bg=bg)
        frames = list(reversed(frames))

        video_path = os.path.join(TARGET_PTH, "receding square", filename)
        write_video(video_path, frames)


    # 3) translating bar: vertical/horizontal translation with contrast calibration
    for filename in translating_bar:
        orientation = "vertical" if "vertical" in filename else "horizontal"
        bg = get_bg_value(filename)
        masks = build_translating_masks(orientation, target_change)
        frames = render_frames_from_masks(masks, bg=bg)

        video_path = os.path.join(TARGET_PTH, "translating bar", filename)
        write_video(video_path, frames)


    # 4) linear deformation: fixed-speed deformation with cross-size calibration
    for filename in linear_deformation:
        mode = "shortening" if "shortening" in filename else "elongating"
        orientation = "horizontal" if "horizontal" in filename else "vertical"
        bg = get_bg_value(filename)
        masks = build_linear_deformation_masks(mode, orientation, target_change)
        frames = render_frames_from_masks(masks, bg=bg)

        video_path = os.path.join(TARGET_PTH, "linear deformation", filename)
        write_video(video_path, frames)

    print(f"Basic test dataset created at:{TARGET_PTH}, with change level calibrated to looming square (mean change: {target_change:.3f})")




if __name__ == "__main__":
    create_basic_test_dataset()



        


