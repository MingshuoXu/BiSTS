import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
import logging
import copy

import json
import torch

from model.bists_lgmd import (BiSTS_LGMD,
                            BiSTS_LGMD_with_Original_EI, BiSTS_LGMD_with_Randomly_Dropout,
                            BiSTS_LGMD_Remove_LTSC,
                            BiSTS_LGMD_Remove_LP_Opt, 
                            BiSTS_LGMD_Adding_G_before_LP, 
                            BiSTS_LGMD_Adding_CN_before_LP, BiSTS_LGMD_Adding_CN_in_PP, 
                            BiSTS_LGMD_Adding_CN_in_PP_and_Adding_G_before_LP, 
                            BiSTS_LGMD_Adding_G_CN_before_LP, BiSTS_LGMD_Adding_CN_G_before_LP,
                            BiSTS_LGMD_abs, 
                            BiSTS_LGMD_abs_I, BiSTS_LGMD_abs_I_Pooling_Conv
                            ) # type: ignore
from effectiveness import effe_noise, effe_contrast  # type: ignore
from lgmd_benchmark.lgmd_benchmark import LgmdBenchmark  # type: ignore
from utils import custom_serialize


BENCHMARK_FOLDER = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 'lgmd_benchmark'))
RAW_DATA_FILE_NAME = os.path.join(os.path.dirname(__file__), file_name_without_ext+'.json')


MODEL_LIST = [
    # baseline
    'BiSTS_LGMD',
    # 1. STSA 
    'with_Original_EI', 'with_Randomly_Dropout', 
    # 2. PSP
    'Remove_LTSC', 
    'Remove_LP_Opt', 
    # 3. Extension
    'Adding_G_before_LP',
    'Adding_CN_before_LP', 'Adding_CN_in_PP', 
    'Adding_CN_in_PP_and_Adding_G_before_LP', 
    'Adding_G_CN_before_LP', 'Adding_CN_G_before_LP',
    # different abs in P layer and STSA
    'BiSTS_LGMD_abs',
    'BiSTS_LGMD_abs_I', 
    'BiSTS_LGMD_abs_I_Pooling_Conv',
    ]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def BiSTS_LGMD_class_api():
    model = BiSTS_LGMD()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_with_Original_EI_class_api():
    model = BiSTS_LGMD_with_Original_EI()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_with_Randomly_Dropout_class_api():
    model = BiSTS_LGMD_with_Randomly_Dropout()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_Remove_LTSC_class_api():
    model = BiSTS_LGMD_Remove_LTSC()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_Remove_LP_Opt_class_api():
    model = BiSTS_LGMD_Remove_LP_Opt()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_Adding_G_before_LP_class_api():
    model = BiSTS_LGMD_Adding_G_before_LP()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_Adding_CN_before_LP_class_api():
    model = BiSTS_LGMD_Adding_CN_before_LP()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_Adding_CN_in_PP_class_api():
    model = BiSTS_LGMD_Adding_CN_in_PP()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_Adding_CN_in_PP_and_Adding_G_before_LP_class_api():
    model = BiSTS_LGMD_Adding_CN_in_PP_and_Adding_G_before_LP()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_Adding_G_CN_before_LP_class_api():
    model = BiSTS_LGMD_Adding_G_CN_before_LP()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_Adding_CN_G_before_LP_class_api():
    model = BiSTS_LGMD_Adding_CN_G_before_LP()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_abs_class_api():
    model = BiSTS_LGMD_abs()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_abs_I_class_api():
    model = BiSTS_LGMD_abs_I()
    model.to(DEVICE)
    return model, 'forward'
def BiSTS_LGMD_abs_I_Pooling_Conv_class_api():
    model = BiSTS_LGMD_abs_I_Pooling_Conv()
    model.to(DEVICE)
    return model, 'forward'


def main(max_workers: int = 8):
    # 创建函数映射字典
    api_functions = {
        # baseline
        'BiSTS_LGMD': BiSTS_LGMD_class_api,
        # 1. STSA
        'with_Original_EI': BiSTS_LGMD_with_Original_EI_class_api,
        'with_Randomly_Dropout': BiSTS_LGMD_with_Randomly_Dropout_class_api,
        # 2. PSP
        'Remove_LTSC': BiSTS_LGMD_Remove_LTSC_class_api,
        'Remove_LP_Opt': BiSTS_LGMD_Remove_LP_Opt_class_api,
        # 3. Extension
        'Adding_G_before_LP': BiSTS_LGMD_Adding_G_before_LP_class_api,
        'Adding_CN_before_LP': BiSTS_LGMD_Adding_CN_before_LP_class_api,
        'Adding_CN_in_PP': BiSTS_LGMD_Adding_CN_in_PP_class_api,
        'Adding_CN_in_PP_and_Adding_G_before_LP': BiSTS_LGMD_Adding_CN_in_PP_and_Adding_G_before_LP_class_api,
        'Adding_G_CN_before_LP': BiSTS_LGMD_Adding_G_CN_before_LP_class_api,
        'Adding_CN_G_before_LP': BiSTS_LGMD_Adding_CN_G_before_LP_class_api,
        'BiSTS_LGMD_abs': BiSTS_LGMD_abs_class_api,
        'BiSTS_LGMD_abs_I': BiSTS_LGMD_abs_I_class_api,
        'BiSTS_LGMD_abs_I_Pooling_Conv': BiSTS_LGMD_abs_I_Pooling_Conv_class_api,
    }


    benchmark_dataset_folder = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                            'lgmd_benchmark'))
    obj_benchmark = LgmdBenchmark(benchmark_dataset_folder)
    for vidname in obj_benchmark.video_list:
        related_path = os.path.relpath(vidname, benchmark_dataset_folder)
        if not related_path.startswith('looming ball against driving sense'):
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
    average_rows = []
    
    num_of_Synthesis = 3244
    num_of_Read_world = 223
    num_of_Noise = (1 + 5*2) * 3
    num_of_Contrast = 3278


    for model in MODEL_LIST:
        noise_rows.append(f'{eval_noise[model]:.1f}')
        contrast_rows.append(f'{eval_contrast[model]:.1f}')
        avg_score = (float(outTable._rows[MODEL_LIST.index(model)][outTable.field_names.index('Synthesis')]) * num_of_Synthesis + \
                     float(outTable._rows[MODEL_LIST.index(model)][outTable.field_names.index('Read-world')]) * num_of_Read_world + \
                     eval_noise[model] * num_of_Noise + \
                     eval_contrast[model] * num_of_Contrast \
                    ) / (num_of_Synthesis + num_of_Read_world + num_of_Noise + num_of_Contrast)

        average_rows.append(f'{avg_score:.1f}')

    outTable.add_column('Noise', noise_rows)
    outTable.add_column('Contrast', contrast_rows)
    outTable.add_column('Average', average_rows)


    outTable.title = "ablation Study"

    print(outTable)

    latex_code = outTable.get_latex_string()

    # 打印出来或者写入文件
    print("\nLaTeX Table Code:\n")
    print(latex_code)


if __name__ == "__main__":
    # main(6) 
    
    evaluate_model()


