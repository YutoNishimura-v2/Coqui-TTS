# 例: qsub_Ag1 -l h_rt='10:00:00' -o ~/logs/yourTTS/20220429_pro_eng_4_202205012016.log run_at_abci.sh
# 例: qsub_Afull -l h_rt='70:00:00' -o ~/logs/yourTTS/20220511_pro_averuni_202205282127.log run_at_abci.sh
# 例: qrsh -g $ABCI_GROUP -l rt_AG.small=1 -l h_rt=12:00:00
# 例: qrsh -g $ABCI_GROUP -l rt_AF=1 -l h_rt=10:00:00
# 例: qrsh -g $ABCI_GROUP -l rt_C.small=1 -l h_rt=100:00:00

source /etc/profile.d/modules.sh
module load gcc/11.2.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
source ~/venv/yourTTS/bin/activate

cd /groups/4/gcd50804/yuto_nishimura/workspace/python/yellston/TTS  # node A
export PYTHONPATH="/groups/4/gcd50804/yuto_nishimura/workspace/python/yellston/TTS:$PYTHONPATH"  # node A

# wandb
export WANDB_API_KEY=372c44d0dd36d935650a41082a67b4ae2cb80015

# # single-GPU
# python3 TTS/bin/train_tts.py \
#     --config_path exps/20220429_pro_eng_4/config.json \
#     --restore_path exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar

# multi-GPU
python3 TTS/bin/distribute.py --script TTS/bin/train_tts.py \
    --config_path exps/20220511_pro_averuni/config.json \
    --restore_path exps/20220425_pro_eng_2/pretrained_model_from_opensource_del_text_emb.pth

deactivate
