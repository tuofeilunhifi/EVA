#!/bin/bash
lizrun start -c "/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/scripts/lizrun_EVA02_CLIP_L_psz14_224to336_on_la1200mwk100m_frozen_internvl_llama_cn_7b.sh" -n 6 -j eva02clipl336-lawk1-2b-fzinllcn7b-6node -i "reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-nlp" #-g 4

