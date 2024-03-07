#!/bin/bash
lizrun start -c "/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/scripts/lizrun_EVA02_CLIP_L_psz14_s4B_on_laion400m_frozen_internvl_llama_cn_7b.sh" -n 8 -j eva02clipl-la400m-fzinllcn7b-8node3 -i "reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-nlp" #-g 4

