#!/bin/bash

# # copy用テンプレ
# rsync -ah --no-i-r --info=progress2 \
#     /mnt/nas/disk1_4tb/dataset/Coefont-Cloud-16khz/ \
#     abci-a_yell:/home/acd14006vc/gcd50804/yuto_nishimura/workspace/python/yellston/dataset/Coefont-Cloud-16khz/

# 一気にepoch系の重みを削除する

target_path="/home/yuto_nishimura/workspace/python/yellston/TTS/checkpoints/20220314_japanese_allial_millial/*/checkpoint_*.pth.tar"

# まずは最大epochの計算
max_epoch=0
for ppath in `\find $target_path -type f`; do
    filename=${ppath##*/}
    if [[ "$filename" == checkpoint_*.pth.tar ]]; then
        epoch=${filename%%.*}
        epoch=${epoch#"checkpoint_"}
        if [ $epoch -gt $max_epoch ]; then
            max_epoch=$epoch
        fi
    fi
done

echo "max epoch is $max_epoch"

# 次に，最大epoch以外の消去
for ppath in `\find $target_path -type f`; do
    filename=${ppath##*/}
    if [[ "$filename" == checkpoint_*.pth.tar ]]; then
        epoch=${filename%%.*}
        epoch=${epoch#"checkpoint_"}
        if [ $epoch -lt $max_epoch ]; then
            rm $ppath
        fi
    fi
done