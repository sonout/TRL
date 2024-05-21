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

### Datasets 

Download San Francisco or Porto Dataset.

* San Francisco (https://ieee-dataport.org/open-access/crawdad-epflmobility)
* Porto (https://www.kaggle.com/datasets/crailtap/taxi-trajectory)



### Preprocessing

For preprocessing use the following notebook:

```
preprocessing
├── porto.ipynb
```

### Training and Evaluation

1. Train Model with 
   ```sh
   python train.py --model <model_name>
   ```
2. Evaluate Model with 
   ```sh
   python evaluate.py --model <model_name> --task <task_name>
   ```

Implemented Models that can be trained and evaluated:
* tigr (TIGR)
* transformer (Transformer)
* t2vec (T2Vec)
* cltsim (CLT-Sim)
* cstte (CSTTE)
* trajcl (TrajCL)
* trembr (Trembr)
* toast (Toast)
* jclm (JCLRNT)
* lightpath (LightPath)
* start (START)

Implemented Downstream Tasks for Evaluation:
* traj_sim (Trajectory Similarity Computation)
* traveltime (Travel Time Estimation)
* destination (Destination Prediction)

