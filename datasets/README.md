### Prepare Data and Pretrained Models

The folder `datasets` should contain files downloaded from [GNN2R - Data and Models](https://osf.io/waqkm/), 
which include our preprocessed datasets and pretrained models for 
WebQuestionsSP, ComplexWebQuestions, PQ-2hop, PQ-3hop, PQL-2hop, and PQL-3hop.

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
