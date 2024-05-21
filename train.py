import torch
import argparse

from pipelines.train import TrainPipeline
from pipelines.evaluate import EvaluatePipeline
from pipelines.utils import load_config

torch.set_float32_matmul_precision('medium')



def main(args):
    print(f'Load and train {args.model}')
    config = load_config(name=args.model, ctype="model")
    config['gpu_device_ids'] = [args.device]
    # Train Pipeline
    pipeline = TrainPipeline(config=config)
    state = pipeline.run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', default='tigr',type=str)
    parser.add_argument('-d','--device', default=1, type=int)

    args = parser.parse_args()

    main(args)