# BiGraph

## Requirements

        pip install -r bigraph_requirements.txt


## Preparing datasets for training

        python preprocess.py --dataset="iemocap_4"

## Training networks 

        python train.py --dataset="iemocap_4" --modalities="atv" --from_begin --epochs=55 --local_global_cat=0  --n_neighbors=6 --wp=1 --wf=1

## Run Evaluation 

        python eval.py --dataset="iemocap_4" --modalities="atv"
