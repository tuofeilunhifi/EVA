#!/bin/bash
lizrun start -c "/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/scripts/lizrun_EVA02_CLIP_B_psz16_s8B_on_coyo_20m_llama2_cn_1.3b.sh" -n 4 -j eva02clipb-coyo20m-llama2cn13b-4node -i "reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-nlp" #-g 4

