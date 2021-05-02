#!/bin/sh -f

script/nirasterscan.py \
	-i data/local/ni2105020336.mkf \
	-p data/ni2105020336.yaml

script/nirasterscan.py \
	-i data/local/maxij1803kp_02.mkf \
	-p data/maxij1803kp_02.yaml

script/nirasterscan.py \
	-i data/local/maxij1803kp_03.mkf \
	-p data/maxij1803kp_03.yaml