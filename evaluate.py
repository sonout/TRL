import torch
import argparse

from pipelines.train import TrainPipeline
from pipelines.evaluate import EvaluatePipeline
from pipelines.utils import load_config

torch.set_float32_matmul_precision('medium')

def main(args):
    model_config = load_config(name=args.model, ctype="model")



    task_config = load_config(name=args.task, ctype="task")
    task_config['gpu_device_ids'] = [args.device]
    eval_pipeline = EvaluatePipeline(model_config=model_config, task_config=task_config)
    res = eval_pipeline.run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', default='tigr',type=str)
    parser.add_argument('-d','--device', default=0, type=int)
    parser.add_argument('-t','--task', default='traj_sim', type=str) # destination, traveltime, traj_sim

    args = parser.parse_args()

    main(args)