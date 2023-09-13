#!/bin/bash
echo "Starting training..."
python -m src.train \
    --gpu_ids='2'\
    --bert_seq_length=128\
    --learning_rate=1e-4\
    --bert_learning_rate=2e-5\
    --savedmodel_path='data/checkpoint/0708_ema_4_stance_twitter_emb_freeze'\
    --max_epochs=4\
    --ema_start=0\
    --ema_decay=0.99\
    --fgm=0\
    --bert_dir="digitalepidemiologylab/covid-twitter-bert"\
    --embedding_freeze
    # --pretrain_model_path="data/pretrain_mlm_nsp_tweet_large/model_epoch_7_loss_1.7978_888.bin"\
    # --premise
    # --gamma_focal=0.001
    # --premise
    # --swa_start=-1\
    # --swa_lr=2e-5
# echo "End training..."