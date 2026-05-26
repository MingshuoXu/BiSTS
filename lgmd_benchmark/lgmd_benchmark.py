import os
import sys
import concurrent.futures
import time

import cv2
from tqdm import tqdm
import numpy as np
import json
from prettytable import PrettyTable
import torch


class LgmdBenchmark():
    def __init__(self, dataset_folder, device='cuda'):
        self.dataset_folder = dataset_folder
        self.device = device

        self.video_list = []
        for root, _, files in os.walk(self.dataset_folder):
            if root.endswith('contrast looming'):
                continue
            for f in files:
                if f.endswith('.mp4') or f.endswith('.avi'):
                    self.video_list.append(os.path.join(root, f))
        self.inference_list = []
        self.benchmark_template = (
            # synthetic scenarios
            "basic test\\looming square",
            "basic test\\receding square",
            "basic test\\translating bar",
            "basic test\\linear deformation",
            "looming ball against driving sense",
            "looming square against nature scenarios",
            # real world scenarios
            "looming ball against static bg",
            "vehicle against driving sense",
            "pure driving sense",
        )

    def run_inference(self, class_api):
        results = {}
        time_record = {}

        for video_path in tqdm(self.inference_list, total=len(self.inference_list), desc="LGMD Benchmark Inference"):
            _res, _time = self._sub_task_inference(class_api, video_path, device=self.device)
            _key = os.path.relpath(video_path, self.dataset_folder).lstrip(os.sep)
            results[_key] = _res
            time_record[_key] = {'time_per_frame': _time}

        return results, time_record
    
    def run_inference_PPE(self, class_api, max_workers=None):
        results = {}
        time_record = {}

        total_processes = os.cpu_count() or 1
        if max_workers is None:
            max_workers = int(total_processes*0.6) if total_processes > 1 else 1
        else:
            max_workers = min(max_workers, total_processes)

        with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
            future_to_video = {executor.submit(self._sub_task_inference, class_api, video_path, self.device): video_path for video_path in self.inference_list}
            for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(self.inference_list), desc="LGMD Benchmark Inference"):
                video_path = future_to_video[future]
                _res, _time = future.result()
                _key = os.path.relpath(video_path, self.dataset_folder).lstrip(os.sep)
                results[_key] = _res
                time_record[_key] = _time
        return results, time_record
    
    def run_inference_seq(self, class_api):
        results = {}
        time_record = {}

        for video_path in tqdm(self.inference_list):
            try:
                _res, _time = self._sub_task_inference(class_api, video_path, device=self.device)
                _key = os.path.relpath(video_path, self.dataset_folder).lstrip(os.sep)
                results[_key] = _res
                time_record[_key] = {'time_per_frame': _time}
            except Exception as exc:
                print(f'Video {video_path} generated an exception: {exc}')
        return results, time_record

    @staticmethod
    def _sub_task_inference(class_api, video_path, device='cuda'):
        class_obj, process_fun_name = class_api()
        # process_func = class_obj.process
        process_func = getattr(class_obj, process_fun_name)
        return inference_single_vid(process_func, video_path, device=device)
    
    def run_evaluation(self, model_outputs):
        
        groundtruth_path = os.path.join(self.dataset_folder, 'annotations.json')
        if not os.path.exists(groundtruth_path):
            raise FileNotFoundError(f"Groundtruth file not found at {groundtruth_path}")
        with open(groundtruth_path, 'r') as f:
            groundtruth_dict = json.load(f)

        all_evaluation = {}
        for key, model_output in model_outputs.items():
            gt = groundtruth_dict.get(key, None)
            if gt is None:
                print(f'Groundtruth not found for video {key}, skipping evaluation.')
                continue
            gt_array = np.asarray(gt['ground_truth'])
            evaluation = calculate_IoU_for_PSP(model_output, gt_array)
            all_evaluation[key] = evaluation

        return all_evaluation
    
    def calculate_distribution_of_dataset(self):
        distribution = {key: 0 for key in self.benchmark_template}
        for video_path in self.video_list:
            rel_path = os.path.relpath(video_path, self.dataset_folder).lstrip(os.sep)
            key1 = os.path.dirname(rel_path)
            if key1 in distribution:
                distribution[key1] += 1

        sum_counts = sum(distribution.values())
        for key in distribution:
            distribution[key] = distribution[key]/sum_counts

        return distribution

    def print_evaluation(self, all_evaluation, model_list=None, is_print=True):

        field_names = ['model', 
                       'basic looming', 'basic receding', 'basic translating', 'basic deformation', 
                       'looming ball', 'looming square', 
                       'static bg ball', 'looming vehicle', 'pure driving'] 
        

        outTable1 = PrettyTable()
        outTable1.field_names = field_names
        outTable1.title = "LGMD Benchmark Evaluation Results with IoU(%)"

        if model_list is None:
            model_list = list(all_evaluation.keys())
        line_datas = []
        weighted_average_separability = {model: 0.0 for model in model_list}
        iou_record = np.zeros((len(model_list), len(self.benchmark_template)))
        dataset_numbers = 0
        for i, model in enumerate(model_list):
            res = all_evaluation[model]
            _benchmark_res_list = {key: [] for key in self.benchmark_template}
            
            model_eval_results = {key: None for key in self.benchmark_template}

            for _filename, value in res.items():
                key1 = os.path.dirname(_filename)
                if key1 in _benchmark_res_list:
                    _benchmark_res_list[key1].append(value)


            for key, value in _benchmark_res_list.items():
                if len(value) > 0:
                    avg_value = sum(value) / len(value)
                    model_eval_results[key] = avg_value
                    weighted_average_separability[model] += sum(value)
                    iou_record[i, self.benchmark_template.index(key)] = avg_value
                    if i == 0:
                        dataset_numbers += len(value)
                else:
                    model_eval_results[key] = None

            line_data = [model, ]
            for j, key in enumerate(self.benchmark_template):
                iou = model_eval_results[key]
                line_data.append(f"{iou*100:.1f}" if iou is not None else '-')


            outTable1.add_row(line_data)
            line_datas.append(line_data)

            weighted_average_separability[model] /= dataset_numbers

        outTable1.add_column('Weighted Average', 
                             [f"{(weighted_average_separability[model]*100):.1f}" for model in model_list]
                             )

        sorted_ranking = sorted(weighted_average_separability.items(), 
                       key=lambda x: x[1], 
                       reverse=True)
        total_rankings = {model: rank+1 for rank, (model, _) in enumerate(sorted_ranking)}
        

        outTable1.add_column('Total Ranking', 
                             [f"{total_rankings[model]:.1f}" for model in model_list]
                             )

        
    
        # outTable 2: synthesis and read-world average
        outTable2 = PrettyTable()
        outTable2.field_names = ['model', 'Synthesis', 'Read-world']
        outTable2.title = "LGMD Benchmark Evaluation Results Summary with IoU(%)"
        dataset_distribution = self.calculate_distribution_of_dataset()
        for i, model in enumerate(model_list):
            synth_avg = 0.0
            realw_avg = 0.0
            total_synth_weight = 0.0
            total_realw_weight = 0.0
            for j in range(6):
                synth_avg += iou_record[i, j]*dataset_distribution[self.benchmark_template[j]]
                total_synth_weight += dataset_distribution[self.benchmark_template[j]]
            synth_avg /= total_synth_weight
            for j in range(6, 9):
                realw_avg += iou_record[i, j]*dataset_distribution[self.benchmark_template[j]]
                total_realw_weight += dataset_distribution[self.benchmark_template[j]]
            realw_avg /= total_realw_weight


            outTable2.add_row([model, f"{synth_avg*100:.1f}", f"{realw_avg*100:.1f}"])

        if is_print:
            print(outTable1)
            # print(outTable2)

        return outTable1, outTable2


def inference_single_vid(process_func, video_path, device='cuda'):
    vid_cap = cv2.VideoCapture(video_path)
    if not vid_cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    results = []
    total_time = 0.0
    frame_count = 0
    while True:
        ret, frame = vid_cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_tensor = torch.from_numpy(gray_frame).float().unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions

        if device == 'cuda':
            torch.cuda.synchronize()  # 确保GPU计算完成
        time_start = time.perf_counter()

        res = process_func(gray_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()  # 确保GPU计算完成
        total_time += time.perf_counter() - time_start

        if isinstance(res, torch.Tensor):
            res = res.item()  # Convert single-value tensor to Python scalar
        results.append(res)

        frame_count += 1

    vid_cap.release()
    avg_time_per_frame = total_time / frame_count if frame_count > 0 else -1
    return results, avg_time_per_frame


def calculate_IoU_for_PSP(modelOpt, groundtruth):
    """
    Calculate Intersection over Union (IoU) for probabilistic score prediction (PSP).
    Args:
        modelOpt: List or array of model outputs (probabilistic scores), ranging from 0.5 to 1
        groundtruth: List or array of ground truth values (probabilistic scores), ranging from 0.5 to 1
    Returns:
        iou: IoU value in the range [0, 1]
    """
    
    # Convert to numpy arrays
    modelOpt = np.asarray(modelOpt)
    groundtruth = np.asarray(groundtruth)

    min_gt = np.min(groundtruth)
    len_gt = len(groundtruth)

    # Calculate intersection and union
    intersection = np.trapezoid(np.minimum(modelOpt, groundtruth) - min_gt) / len_gt * 2 
    union = np.trapezoid(np.maximum(modelOpt, groundtruth) - min_gt) / len_gt * 2 

    if np.max(groundtruth) == 0.5:
        return 1 - union
    else:
        # Prevent division by zero
        if union == 0:
            return 0.0
        else:
            return intersection / union
    

def custom_serialize(obj, indent=2, current_level=0):
    if isinstance(obj, list):
        current_level += 1
        if any(isinstance(item, (list, dict)) for item in obj):
            # 包含嵌套结构，不是最后一级
            indent_str = ' ' * (indent * current_level)
            items = [f"{indent_str}{custom_serialize(item, indent, current_level)}" for item in obj]
            outer_indent = ' ' * (indent * (current_level - 1)) if current_level > 1 else ''
            return '[\n' + ',\n'.join(items) + '\n' + outer_indent + ']'
        else:
            # 最后一级，都是基本类型
            return '[' + ', '.join(json.dumps(item, ensure_ascii=False) for item in obj) + ']'
    
    elif isinstance(obj, dict):
        current_level += 1
        indent_str = ' ' * (indent * current_level)
        items = []
        for key, value in obj.items():
            serialized_value = custom_serialize(value, indent, current_level)
            items.append(f'{indent_str}{json.dumps(key, ensure_ascii=False)}: {serialized_value}')
        
        outer_indent = ' ' * (indent * (current_level - 1)) if current_level > 1 else ''
        return '{\n' + ',\n'.join(items) + '\n' + outer_indent + '}'
    
    else:
        return json.dumps(obj, ensure_ascii=False)
    