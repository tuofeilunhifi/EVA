#!/bin/bash
lizrun start -c "/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/scripts/lizrun_EVA02_CLIP_L_psz14_s4B_on_wukong100m_laion400m_llama2_tokenizer.sh" -n 16 -j eva02clipl-wkla-llama2tk-16node -i "reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-nlp" #-g 4

