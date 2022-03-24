jpとその他の言語で音素数が異なる．
jpは音素に崩す前の長さで計算されてしまうため，
「min_seq_len」が同じだと狂う．

なので，ルールベースで以下の文言を追加した

> if lang == "ja-jp":
    _min_seq_len = 11
else:
    _min_seq_len = self.min_seq_len
