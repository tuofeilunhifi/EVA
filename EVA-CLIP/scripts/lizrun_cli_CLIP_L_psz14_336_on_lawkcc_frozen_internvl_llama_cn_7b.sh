#!/bin/bash
lizrun start -c "/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/scripts/lizrun_CLIP_L_psz14_336_on_lawkcc_frozen_internvl_llama_cn_7b.sh" -n 10 -j clipl336-lawkcc-fzinllcn7b-10node -i "reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-nlp" # -p detect #-g 4

