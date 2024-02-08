#!/bin/bash
lizrun start -c "/mnt/pfs-guan-ssai/cv/cjy/codebase/EVA/EVA-CLIP/scripts/lizrun_EVA02_CLIP_E_psz14_plus_s9B_on_coyo_20m_llama_cn_7b.sh" -n 8 -j eva02-clipep-coyo20m-llamacn7b-8node -i "reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.0-multinode-nlp" #-g 4

