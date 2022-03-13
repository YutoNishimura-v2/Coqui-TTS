# 例: qsub_Ag1 -l h_rt='150:00:00' -o ~/logs/yourTTS/202203_japanese+eng+fr_202203140530.log run_at_abci.sh
# 例: qrsh -g $ABCI_GROUP -l rt_AG.small=1 -l h_rt=10:00:00

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 cuda/11.1/11.1.1 cudnn/8.0/8.0.5
source ~/venv/yourTTS/bin/activate

# 具体的処理
cd /groups/4/gcd50804/yuto_nishimura/TTS  # node A
export PYTHONPATH="/groups/4/gcd50804/yuto_nishimura/TTS:$PYTHONPATH"  # node A

python3 TTS/bin/train_tts.py \
    --config_path exps/202203_japanese+eng+fr/config.json \
    --restore_path exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar
deactivate
