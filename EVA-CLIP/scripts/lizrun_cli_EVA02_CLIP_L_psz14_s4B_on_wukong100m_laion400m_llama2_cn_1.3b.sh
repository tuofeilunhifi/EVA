#!/bin/bash
lizrun start -c "/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/scripts/lizrun_EVA02_CLIP_L_psz14_s4B_on_wukong100m_laion400m_llama2_cn_1.3b.sh" -n 8 -j eva02clipl-wkla400m-llama2cn13b-8node -i "reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-nlp" #-g 4

