# GNN2R: Weakly-Supervised Rationale-Providing Question Answering over Knowledge Graphs

This repository includes the code for our paper **GNN2R: Weakly-Supervised Rationale-Providing Question Answering over Knowledge Graphs**.

The link to the paper and citation information will be added soon. 

Feel free to contact [me](https://www.ifi.uzh.ch/en/ddis/people/ruijie.html) if there is any questions.

### Environment

Only the following packages need to be manually installed in the environment ([conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)): 
[python 3.10.12](https://www.python.org/downloads/release/python-31012/),  [graph-tool 2.58](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions),
[pytorch 2.0.1](https://pytorch.org/get-started/locally/), [transformers 4.32.1](https://huggingface.co/docs/transformers/installation),
[pytorch-scatter 2.1.1](https://github.com/rusty1s/pytorch_scatter), [tqdm 4.66.1](https://tqdm.github.io).

### Prepare Data and Pretrained Models

The folder `datasets` should contain files downloaded from [GNN2R - Data and Models](https://osf.io/waqkm/), 
which include our preprocessed datasets and pretrained models for 
**WebQuestionsSP**, **ComplexWebQuestions**, **PQ-2hop**, **PQ-3hop**, **PQL-2hop**, and **PQL-3hop**.

The structure of this folder:

`datasets`
  * [dataset name] -- e.g., `WebQuestionsSP`
    * `in_path`
      * [preprocessed data dumps] -- e.g., `train_qid2que.pickle`
    * `out_path`
      * [timestamp of a pre-trained model for GNN2R Step-I] -- e.g., `2022.11.21.17.13.23`
        * `best.tar`-- the pre-trained model (Step-I)
        * [M-N] -- for training the model below, top-M and top-N answers were considered in Step-II during training and validation/test, respectively, e.g., `50-10`
          * [preprocessed data dumps for Step-II] -- e.g, `train_qid2subg_exp.pickle`
          * [timestamp of a pre-trained model for GNN2R Step-II] -- e.g., `2023.01.18.21.17.49`
            * `best.tar`- the pre-trained model (Step-II)

* If you are interested in the original data, please refer to [NSM](https://github.com/RichardHGL/WSDM2021_NSM) (WebQuestionsSP and ComplexWebQuestions) and [IRN](https://github.com/zmtkeke/IRN) (PQ and PQL).
* We used `que_prep.py` to preprocess WebQuestionsSP and CWQ, and `que_prep_pq.py` to preprocess PQ and PQL.

-----

### Directly Apply Our Pretrained Models in Evaluation

`main.py` and `subgraph_reasoning.py` (`subgraph_reasoning_pq.py` for PQ and PQL) are the scripts for Step-I and Step-II of the GNN2R model, respectively. 

The following commands can be used to load and evaluate our pre-trained models on respective test sets.

#### WebQuestionsSP (WQSP)
  ```shell
    python -u main.py --dataset WebQuestionsSP --pre_timestamp 2022.11.21.17.13.23 --num_epochs 0 --threshold 1.0
  ```

  ```shell
  python -u subgraph_reasoning.py --dataset WebQuestionsSP --align_timestamp 2022.11.21.17.13.23 --pre_timestamp 2023.01.18.21.17.49 --num_epochs 0 --train_top 50 --valid_test_top 10 --threshold 1.2
  ```

#### ComplexWebQuestions (CWQ)
  ```shell
  python -u main.py --dataset CWQ --pre_timestamp 2023.01.20.14.35.51 --num_epochs 0 --threshold 1.0
  ```
  
  ```shell
  python -u subgraph_reasoning.py --dataset CWQ --align_timestamp 2023.01.20.14.35.51 --pre_timestamp 2023.01.23.16.10.23 --num_epochs 0 --train_top 25 --valid_test_top 25 --max_num_subgs_per_ent 1000 --threshold 1.2
  ```
  
#### PQ-2hop
  ```shell
  python -u main.py --dataset pq-2hop --pre_timestamp 2023.09.06.23.14.07 --num_epochs 0 --threshold 1.0
  ```
  
  ```shell
  python -u subgraph_reasoning_pq.py --dataset pq-2hop --align_timestamp 2023.09.06.23.14.07 --pre_timestamp 2023.09.10.19.32.26 --num_epochs 0 --train_top 20 --valid_test_top 20 --threshold 1.2
  ```

#### PQ-3hop
  ```shell
  python -u main.py --dataset pq-3hop --pre_timestamp 2023.09.07.03.15.23 --num_epochs 0 --threshold 1.0
  ```
  
  ```shell
  python -u subgraph_reasoning_pq.py --dataset pq-3hop --align_timestamp 2023.09.07.03.15.23 --pre_timestamp 2023.09.10.23.46.59 --cutoff 3 --num_epochs 0 --train_top 20 --valid_test_top 5 --threshold 1.2
  ```

#### PQL-2hop
  ```shell
  python -u main.py --dataset pql-2hop --pre_timestamp 2023.09.06.22.02.06 --num_epochs 0 --threshold 1.0
  ```
  
  ```shell
  python -u subgraph_reasoning_pq.py --dataset pql-2hop --align_timestamp 2023.09.06.22.02.06 --pre_timestamp 2023.09.10.20.00.30 --num_epochs 0 --train_top 20 --valid_test_top 5 --threshold 1.05
  ```

#### PQL-3hop
  ```shell
  python -u main.py --dataset pql-3hop --pre_timestamp 2023.09.07.02.43.20 --num_epochs 0 --threshold 1.0
  ```
  
  ```shell
  python -u subgraph_reasoning_pq.py --dataset pql-3hop --align_timestamp 2023.09.07.02.43.20 --pre_timestamp 2023.09.08.18.18.38 --cutoff 3 --num_epochs 0 --train_top 20 --valid_test_top 20 --threshold 1.2
  ```
-----

### Train New Models

The following commands can be used for training new models. The configurations are those we used in experiments.

Again, `main.py` and `subgraph_reasoning.py` (`subgraph_reasoning_pq.py` for PQ and PQL) are the scripts for Step-I and Step-II, respectively. 

Please take a note of the timestamp of the Step-I model trained by `main.py` and fill it in the placeholder `[timestamp]` below for running `subgraph_reasoning.py` or `subgraph_reasoning_pq.py`.

#### WebQuestionsSP (WQSP)
  ```shell
  python -u main.py --dataset WebQuestionsSP --num_epochs 50 --num_gcn_layers 3 --batch_size 16 --lr 0.0005 --weight_decay 1e-05 --dropout 0.0 --margin 1.0
  ```
  
  ```shell
  python -u subgraph_reasoning.py --dataset WebQuestionsSP --align_timestamp [timestamp] --train_top 50 --valid_test_top 10 --prep_subg --margin 0.1 --eval_enc_batch 2048 --num_epochs 50 --batch_size 16 --neg_size 32 --lr 8e-06 --lr_decay 1.0 --weight_decay 0.001 --num_warmup_steps 100 --reinit_n 0 --valid_batch_freq 80
  ```

#### ComplexWebQuestions (CWQ)
  ```shell
  python -u main.py --dataset CWQ --num_epochs 50 --num_gcn_layers 3 --batch_size 24 --lr 0.0005 --weight_decay 1e-05 --dropout 0.0 --margin 0.5`
  ```
  
  ```shell
  python -u subgraph_reasoning.py --dataset CWQ --align_timestamp [timestamp] --train_top 25 --valid_test_top 25 --prep_subg --margin 0.1 --eval_enc_batch 512 --num_epochs 50 --batch_size 12 --neg_size 32 --lr 8e-06 --lr_decay 1.0 --weight_decay 0.001 --num_warmup_steps 500 --reinit_n 0 --valid_batch_freq 500`
  ```

#### PQ-2hop
  ```shell
  python -u main.py --dataset pq-2hop --num_epochs 100 --num_gcn_layers 3 --batch_size 16 --lr 0.0005 --weight_decay 1e-05 --dropout 0.0 --in_dim 768 --hid_dim 256 --norm 2 --margin 0.5 --loss_red mean --max_pos_neg_pairs 50000
  ```
  
  ```shell
  python -u subgraph_reasoning_pq.py --align_timestamp [timestamp] --dataset pq-2hop --lm_name sentence-transformers/multi-qa-distilbert-cos-v1 --train_top 20 --valid_test_top 20 --prep_subg --cutoff 2 --in_dim 768 --hid_dim 256 --align_hid_dim 256 --num_gcn_layers 3 --norm 2 --margin 0.8 --eval_enc_batch 512 --num_epochs 50 --batch_size 12 --neg_size 32 --lr 8e-06 --lr_decay 1.0 --weight_decay 0.001 --num_warmup_steps 500 --reinit_n 0 --valid_batch_freq 500 --loss_red mean --max_num_subgs_per_ent 10000
  ```
 
#### PQ-3hop
  ```shell
  python -u main.py --dataset pq-3hop --num_epochs 100 --num_gcn_layers 3 --batch_size 16 --lr 0.0005 --weight_decay 1e-05 --dropout 0.0 --in_dim 768 --hid_dim 256 --norm 2 --margin 1.0 --loss_red mean --max_pos_neg_pairs 50000
  ```

  ```shell
  python -u subgraph_reasoning_pq.py --align_timestamp [timestamp] --dataset pq-3hop --lm_name sentence-transformers/multi-qa-distilbert-cos-v1 --train_top 20 --valid_test_top 5 --prep_subg --cutoff 3 --in_dim 768 --hid_dim 256 --align_hid_dim 256 --num_gcn_layers 3 --norm 2 --margin 0.1 --eval_enc_batch 512 --num_epochs 50 --batch_size 12 --neg_size 32 --lr 8e-06 --lr_decay 1.0 --weight_decay 0.001 --num_warmup_steps 500 --reinit_n 0 --valid_batch_freq 500 --loss_red mean --max_num_subgs_per_ent 10000
  ```
 
#### PQL-2hop
  ```shell
  python -u main.py --dataset pql-2hop --num_epochs 100 --num_gcn_layers 3 --batch_size 16 --lr 0.0005 --weight_decay 1e-05 --dropout 0.0 --in_dim 768 --hid_dim 256 --norm 2 --margin 1.0 --loss_red mean --max_pos_neg_pairs 50000
  ```
  
  ```shell
  python -u subgraph_reasoning_pq.py --align_timestamp [timestamp] --dataset pql-2hop --lm_name sentence-transformers/multi-qa-distilbert-cos-v1 --train_top 20 --valid_test_top 5 --prep_subg --cutoff 2 --in_dim 768 --hid_dim 256 --align_hid_dim 256 --num_gcn_layers 3 --norm 2 --margin 0.8 --eval_enc_batch 512 --num_epochs 50 --batch_size 12 --neg_size 32 --lr 8e-06 --lr_decay 1.0 --weight_decay 0.001 --num_warmup_steps 500 --reinit_n 0 --valid_batch_freq 500 --loss_red mean --max_num_subgs_per_ent 10000
  ```

#### PQL-3hop
  ```shell
  python -u main.py --dataset pql-3hop --num_epochs 100 --num_gcn_layers 3 --batch_size 16 --lr 0.0005 --weight_decay 1e-05 --dropout 0.0 --in_dim 768 --hid_dim 256 --norm 2 --margin 1.0 --loss_red mean --max_pos_neg_pairs 50000
  ```
  
  ```shell
  python -u subgraph_reasoning_pq.py --align_timestamp 2023.12.01.16.05.22 --dataset pql-3hop --lm_name sentence-transformers/multi-qa-distilbert-cos-v1 --train_top 20 --valid_test_top 20 --prep_subg --cutoff 3 --in_dim 768 --hid_dim 256 --align_hid_dim 256 --num_gcn_layers 3 --norm 2 --margin 0.05 --eval_enc_batch 512 --num_epochs 50 --batch_size 12 --neg_size 32 --lr 8e-06 --lr_decay 1.0 --weight_decay 0.001 --num_warmup_steps 500 --reinit_n 0 --valid_batch_freq 500 --loss_red mean --max_num_subgs_per_ent 10000
  ```