import os
import sys
itemPth = os.path.dirname(os.path.dirname(__file__))
sys.path.append(itemPth)
file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]

import json
import torch

from model.bists_lgmd import BiSTS_LGMD # type: ignore
from model.existing_lgmd import (BasicLGMD, pLGMD, LGMD_P_ON_OFF,
                                 SFA_LGMD, 
                                 LGMD1_LGMD2_CascadeNetwork_Dual_Channel,
                                 LGMD2_LGMD1_CascadeNetwork_Dual_Channel,
                                 ALGMD, EMD_LPLC2_GF
                                ) # type: ignore
from lgmd_benchmark import LgmdBenchmark, custom_serialize # type: ignore

RAW_DATA_FILE_NAME = os.path.join(os.path.dirname(__file__), file_name_without_ext.replace('run_', 'raw_')+'.json')


MODEL_LIST = [
              'BiSTS_LGMD',
              'BasicLGMD', 'LGMD_P_ON_OFF', 'pLGMD', 
              'SFA_LGMD',
              'LGMD1_LGMD2_CascadeNetwork_Dual_Channel', 'LGMD2_LGMD1_CascadeNetwork_Dual_Channel',
              'ALGMD', 
              'EMD_LPLC2_GF'
              ]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def BiSTS_LGMD_class_api():
    model = BiSTS_LGMD()
    model.to(DEVICE)
    return model, 'forward'
def BasicLGMD_class_api():
    model = BasicLGMD()
    model.to(DEVICE)
    return model, 'forward'
def pLGMD_class_api():
    model = pLGMD()
    model.to(DEVICE)
    return model, 'forward'
def LGMD_P_ON_OFF_class_api():
    model = LGMD_P_ON_OFF()
    model.to(DEVICE)
    return model, 'forward'
def SFA_LGMD_class_api():
    model = SFA_LGMD()
    model.to(DEVICE)
    return model, 'forward'
def LGMD1_LGMD2_CascadeNetwork_Dual_Channel_class_api():
    model = LGMD1_LGMD2_CascadeNetwork_Dual_Channel()
    model.to(DEVICE)
    return model, 'forward'
def LGMD2_LGMD1_CascadeNetwork_Dual_Channel_class_api():
    model = LGMD2_LGMD1_CascadeNetwork_Dual_Channel()
    model.to(DEVICE)
    return model, 'forward'
def ALGMD_class_api():
    model = ALGMD()
    model.to(DEVICE)
    return model, 'forward'
def EMD_LPLC2_GF_class_api():
    model = EMD_LPLC2_GF()
    model.to(DEVICE)
    return model, 'forward'




def inference(model_name):
    benchmark_dataset_folder = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                            'lgmd_benchmark'))
    benchmark = LgmdBenchmark(benchmark_dataset_folder)
    benchmark.inference_list = benchmark.video_list  # Use all videos in the folder
    
    print(f'Running benchmark for model: {model_name}')
    class_api = getattr(sys.modules[__name__], f'{model_name}_class_api')
    results, time_record = benchmark.run_inference(class_api)

    if os.path.exists(RAW_DATA_FILE_NAME):
        try:
            with open(RAW_DATA_FILE_NAME, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = {}
    else:
        existing_data = {}

    existing_data[model_name] = {'results': results, 'time_record': time_record}

    save_file = custom_serialize(existing_data, indent=2)
    with open(RAW_DATA_FILE_NAME, 'w') as f:
        f.write(save_file)

    print(f'Benchmark results saved to {RAW_DATA_FILE_NAME}')


def evaluation():
    with open(RAW_DATA_FILE_NAME, 'r') as f:
        data = json.load(f)
    
    eval_res = {}
    for model, res in data.items():
        # print(f'Evaluating benchmark for model: {model}')
        benchmark_dataset_folder = os.path.abspath(os.path.join(itemPth, '..', '7_Dataset', 
                                                                'lgmd_benchmark'))
        benchmark = LgmdBenchmark(benchmark_dataset_folder)
        benchmark.inference_list = benchmark.video_list  # Use all videos in the folder

        eval_res[model] = benchmark.run_evaluation(res['results'])

    benchmark.print_evaluation(eval_res, model_list=MODEL_LIST)


def main():
    for model in MODEL_LIST:
        inference(model)


if __name__ == '__main__':
    # main()

    evaluation()


 