#!/bin/bash
lizrun start -c "/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/scripts/lizrun_CLIP_B_psz16_s4B_on_lawkcc_frozen_internvl_llama_cn_7b.sh" -n 8 -j clipb-lawkcc-fzinllcn7b-8node -i "reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-nlp" # -p detect #-g 4

