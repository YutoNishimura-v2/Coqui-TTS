# 例: qsub_Ag1 -l h_rt='150:00:00' -o ~/logs/yourTTS/20220417_pro_eng_singleGPU_batch_52_202204192301.log run_at_abci.sh
# 例: qsub_Afull -l h_rt='70:00:00' -o ~/logs/yourTTS/20220425_pro_eng_2_202204290717.log run_at_abci.sh
# 例: qrsh -g $ABCI_GROUP -l rt_AG.small=1 -l h_rt=10:00:00
# 例: qrsh -g $ABCI_GROUP -l rt_AF=1 -l h_rt=10:00:00

source /etc/profile.d/modules.sh
module load gcc/11.2.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
source ~/venv/yourTTS/bin/activate

cd /groups/4/gcd50804/yuto_nishimura/workspace/python/yellston/TTS  # node A
export PYTHONPATH="/groups/4/gcd50804/yuto_nishimura/workspace/python/yellston/TTS:$PYTHONPATH"  # node A

# # single-GPU
# python3 TTS/bin/train_tts.py \
#     --config_path exps/20220417_pro_eng/config.json \
#     --restore_path exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar

# # multi-GPU
python3 TTS/bin/distribute.py --script TTS/bin/train_tts.py \
    --config_path exps/20220425_pro_eng_2/config.json \
    --restore_path checkpoints/20220425_pro_eng_2/vits_tts-portuguese-April-27-2022_12+43AM-0466a642/checkpoint_70000.pth.tar

deactivate
