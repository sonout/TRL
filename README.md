<!-- ABOUT THE PROJECT -->
<!-- ## About The Project -->
# Trajectory Representation Learning on Road Networks and Grids

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
To install the required packages we recommend using [Conda](https://docs.conda.io/en/latest/). Our used environment can be easily installed with conda.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/...
   ```
2. Install conda environment
   ```sh
   conda config --env --set channel_priority strict
   conda env create -f environment.yml
   ```
3. Activate the environment
   ```sh
   conda activate traj-emb
   ```

<!-- USAGE EXAMPLES -->


## Usage

1. Preprocess Data with notebook in preprocessing/porto.ipynb 

2. Train Model with 
   ```sh
   python train.py --model <model_name>
   ```
3. Evaluate Model with 
   ```sh
   python evaluate.py --model <model_name> --task <tsk_name>
   ```
