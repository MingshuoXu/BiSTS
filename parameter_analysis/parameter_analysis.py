import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]

import json
from prettytable import PrettyTable
import copy
import numpy as np

from model.bists_lgmd import (BiSTS_LGMD, BiSTS_LGMD_N,
                            BiSTS_LGMD_with_STSA_remove_Conv,
                            BiSTS_LGMD_Pooling_before_Conv,
                            ) # type: ignore
from effectiveness import effe_noise, effe_contrast  # type: ignore
from lgmd_benchmark.lgmd_benchmark import LgmdBenchmark  # type: ignore
from utils import custom_serialize
import logging

BENCHMARK_FOLDER = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                  'lgmd_benchmark'))

RAW_DATA_FILE_NAME = os.path.join(os.path.dirname(__file__), 
                                  file_name_without_ext+'.json')


MODEL_LIST = [
              'BiSTS_LGMD', 
              'with_size_3', 'with_size_5',
              'with_stride_2',
              'with_size_3_stride_2',
              'with_size_5_stride_2',
              'with_size_5_stride_3',
              'with_STSA_remove_Conv', 
              'Pooling_before_Conv', 
              ]


def BiSTS_LGMD_class_api():
    model = BiSTS_LGMD()
    model.to('cuda')
    return model, 'forward'
def BiSTS_LGMD_with_size_3_class_api():
    model = BiSTS_LGMD()
    model.STSA.poolingSize = (3, 3)
    model.to('cuda')
    return model, 'forward'
def BiSTS_LGMD_with_size_5_class_api():
    model = BiSTS_LGMD()
    model.STSA.poolingSize = (5, 5)
    model.to('cuda')
    return model, 'forward'
def BiSTS_LGMD_with_stride_2_class_api():
    model = BiSTS_LGMD()
    model.STSA.stride = (2, 2)
    model.to('cuda')
    return model, 'forward'
def BiSTS_LGMD_N_class_api():
    model = BiSTS_LGMD_N()
    model.to('cuda')
    return model, 'forward'
def BiSTS_LGMD_with_size_5_stride_2_class_api():
    model = BiSTS_LGMD()
    model.STSA.poolingSize = (5, 5)
    model.STSA.stride = (2, 2)
    model.to('cuda')
    return model, 'forward'
def BiSTS_LGMD_with_size_5_stride_3_class_api():
    model = BiSTS_LGMD()
    model.STSA.poolingSize = (5, 5)
    model.STSA.stride = (3, 3)
    model.to('cuda')
    return model, 'forward'
def BiSTS_LGMD_with_STSA_remove_Conv_class_api():
    model = BiSTS_LGMD_with_STSA_remove_Conv()
    model.to('cuda')
    return model, 'forward'
def BiSTS_LGMD_Pooling_before_Conv_class_api():
    model = BiSTS_LGMD_Pooling_before_Conv()
    model.to('cuda')
    return model, 'forward'


def main(max_workers=2):
    # 创建函数映射字典
    api_functions = {
        # baseline
        'BiSTS_LGMD': BiSTS_LGMD_class_api,
        # 1 pooling size = 3
        'with_size_3': BiSTS_LGMD_with_size_3_class_api,
        # 2 pooling size = 5
        'with_size_5': BiSTS_LGMD_with_size_5_class_api,
        # 3 stride = 2
        'with_stride_2': BiSTS_LGMD_with_stride_2_class_api,
        # 4 pooling size = 3 and stride = 2
        'with_size_3_stride_2': BiSTS_LGMD_N_class_api,
        'with_size_5_stride_2': BiSTS_LGMD_with_size_5_stride_2_class_api,
        'with_size_5_stride_3': BiSTS_LGMD_with_size_5_stride_3_class_api,
        # 5 remove Conv layer after STSA
        'with_STSA_remove_Conv': BiSTS_LGMD_with_STSA_remove_Conv_class_api,
        # 6 pooling before Conv layer
        'Pooling_before_Conv': BiSTS_LGMD_Pooling_before_Conv_class_api,
    }


    benchmark_dataset_folder = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                            'lgmd_benchmark'))
    obj_benchmark = LgmdBenchmark(benchmark_dataset_folder)
    for vidname in obj_benchmark.video_list:
        related_path = os.path.relpath(vidname, benchmark_dataset_folder)
        if not related_path.startswith('looming ball against driving sense'):
        # if related_path.startswith('basic test') and not related_path.startswith('basic test\\contrast looming'):
            obj_benchmark.inference_list.append(vidname)
    
    for model_name in MODEL_LIST:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Evaluating model: %s", model_name)
        class_api = api_functions[model_name]

        noise_res = effe_noise.inference_model(class_api, max_workers=max_workers)
        contrast_res = effe_contrast.inference_model(class_api, max_workers=max_workers)
        benchmark_res, _ = obj_benchmark.run_inference_PPE(class_api, max_workers=max_workers)

        for key in contrast_res.keys():
            if 'looming_ball_against_driving_sense' in key:
                contrast_res_looming_ball_against_driving_sense = contrast_res[key]
                break

        for key, value in contrast_res_looming_ball_against_driving_sense.items():
            benchmark_res[f'looming ball against driving sense\\{key}'] = value


        results = {
            'noise': noise_res,
            'contrast': contrast_res,
            'benchmark': benchmark_res,
        }

        if os.path.exists(RAW_DATA_FILE_NAME):
            with open(RAW_DATA_FILE_NAME, 'r') as f:
                existing_data = json.load(f)
                if len(existing_data) == 0:
                    existing_data = {}
        else:
            existing_data = {}

        existing_data[model_name] = results

        save_file = custom_serialize(existing_data, indent=2)
        with open(RAW_DATA_FILE_NAME, 'w') as f:
            f.write(save_file)


def evaluate_model(is_print = True):
    with open(RAW_DATA_FILE_NAME, 'r') as f:
        results = json.load(f)

    noise_res = {model: results[model]['noise'] for model in MODEL_LIST}
    contrast_res = {model: results[model]['contrast'] for model in MODEL_LIST}

    eval_noise = effe_noise.comparison_performance(noise_res, models=MODEL_LIST, is_print=is_print)
    eval_contrast = effe_contrast.comparison_performance(contrast_res, models=MODEL_LIST, is_print=is_print)

    benchmark_dataset_folder = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                            'lgmd_benchmark'))
    obj_benchmark = LgmdBenchmark(benchmark_dataset_folder)
    obj_benchmark.inference_list = obj_benchmark.video_list
    
    eval_res = {}
    for model, res in results.items():
        eval_res[model] = obj_benchmark.run_evaluation(res['benchmark'])

    _, eval_benchmark = obj_benchmark.print_evaluation(eval_res, model_list=MODEL_LIST, is_print=is_print)

    outTable = copy.deepcopy(eval_benchmark)

    noise_rows = []
    contrast_rows = []
    
    for model in MODEL_LIST:
        noise_rows.append(f'{eval_noise[model]:.1f}')
        contrast_rows.append(f'{eval_contrast[model]:.1f}')

    outTable.add_column('Noise', noise_rows)
    outTable.add_column('Contrast', contrast_rows)

    outTable.title = "ablation Study with IoU(%)"

    print(outTable)


if __name__ == "__main__":
    # main(6) 

    evaluate_model()


