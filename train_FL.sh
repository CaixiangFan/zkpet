#!/bin/bash

sites=( 0  2   4   6   9  11  12  25  29  36  46  50  57
  59  68  70  75  81  83  84  85  86  87  97  98  99
  103 104 105 106 107 111 113 114 115 116 117 118 119
  120 121 122 123 124 125 126 127 128 129 130 131 132
  133 146 182 245 270 298 299 303 310 317 318 319 320 )

model_type=Linear
seq_len=192
for site in "${sites[@]}"
do
  # Run client with args: model_type, target, seq_len 
  echo training site $site
  python client_LTSF.py $model_type $site $seq_len &
done >> logs/fl_train.log