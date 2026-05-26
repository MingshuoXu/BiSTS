import argparse
import os
import sys
ITEM_PTH = os.path.dirname(os.path.dirname(__file__))
if ITEM_PTH not in sys.path:
    sys.path.append(ITEM_PTH)

import json
import torch
# import torch.nn as nn
# import torch.nn.functional as F

from lgmd_benchmark import LgmdBenchmark, custom_serialize


# ==========================================
# Step 1: Define your custom model class (or classes)
# ==========================================
class MyCustomLGMDModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Initialize your model weights and components here
        # self.model = ... 

    def forward(self, gray_tensor):
        """
        Core inference function.
        Input: gray_tensor, a grayscale image Tensor with shape [1, 1, H, W]
        Output: a scalar float (typically between 0.5 and 1.0 representing collision probability)
        """
        # --- Replace with the real inference logic ---
        # Example: simulate outputting a random number between 0.5 and 1.0
        import random
        prediction = 0.5 + random.random() * 0.5 
        return prediction


# ==========================================
# Step 2: Define the API wrapper for Benchmark to call
# ==========================================
def custom_model_api(device='cuda'):
    """
    The LGMD Benchmark requires an API function that returns (model_instance, 'inference_method_name').
    """
    model = MyCustomLGMDModel()
    # model.to(device)  # If your model has a .to() method, uncomment this line to move it to the correct device.

    return model, 'forward'


# ==========================================
# Step 3: Run the benchmark pipeline
# ==========================================
def main():
    '''
        ```bash
        python eval_custom_model.py --dataset "/path/to/lgmd_benchmark" --model_name "MySuperLGMD"
        ```
    '''
    parser = argparse.ArgumentParser(description="Evaluate a custom model on the LGMD Benchmark")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the lgmd_benchmark dataset folder")
    parser.add_argument('--model_name', type=str, default="MyCustomModel", help="Name of your model for the evaluation table")
    parser.add_argument('--output', type=str, default="benchmark_results.json", help="Path to save raw inference results")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the model on (default: auto-detect)")
    parser.add_argument('--use_ppe', action='store_true', help="Use Parallel Process Executor for inference (if supported by your model)")
    args = parser.parse_args()

    dataset_folder = os.path.abspath(args.dataset)
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Dataset folder not found at: {dataset_folder}")

    # 1. Initialize Benchmark
    benchmark = LgmdBenchmark(dataset_folder, device=args.device)
    benchmark.inference_list = benchmark.video_list

    print(f"\n--- [1/2] Running Inference for {args.model_name} ---")
    # 2. Run inference (you can use run_inference_PPE for parallel inference)
    if args.use_ppe:
        print("Using Parallel Process Executor for inference...")
        results, time_record = benchmark.run_inference_PPE(custom_model_api)
    else:
        results, time_record = benchmark.run_inference(custom_model_api)

    # 3. Save raw results to JSON
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                all_data = json.load(f)
        except json.JSONDecodeError:
            all_data = {}
    else:
        all_data = {}

    all_data[args.model_name] = {'results': results, 'time_record': time_record}
    
    with open(os.path.join(os.path.dirname(__file__), 'raw_benchmark.json'), 'r') as f:
        raw_benchmark_data = json.load(f)
    for key in raw_benchmark_data:
        if key not in all_data.keys():
            all_data[key] = raw_benchmark_data[key]

    with open(args.output, 'w') as f:
        f.write(custom_serialize(all_data, indent=2))
    print(f"Raw results saved to {args.output}")

    # 4. Run evaluation and print the table
    print(f"\n--- [2/2] Running Evaluation for {args.model_name} ---")
    eval_res = {key: benchmark.run_evaluation(val['results']) for key, val in all_data.items()}
    benchmark.print_evaluation(eval_res)


if __name__ == '__main__':
    main()