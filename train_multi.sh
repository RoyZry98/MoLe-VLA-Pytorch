TASK=task3
FUTURE_ACTION_STEPS=15
skip_layer_number=$1
add_lstm=$2
route_in_for=$3
use_index=$4 
seed=$5
ema_decay=$6
# SETTING=freeze_vit_window${FUTURE_ACTION_STEPS}
SETTING=Attn_weight_KD_ema_decay_${ema_decay}_seed_${seed}_skip_layer_${skip_layer_number}_lstm_architecture_${add_lstm}_freeze_none_window${FUTURE_ACTION_STEPS}_${TASK}_route_in_for_${route_in_for}
# SETTING=index_strategy_freeze_none_window${FUTURE_ACTION_STEPS}_Llama2_MoE_bz_4_new_DiT_skip_llama_layer_${skip_layer_number}_lstm_architecture_${add_lstm}_${TASK}_route_in_for_${route_in_for}
# SETTING=freeze_none_window${FUTURE_ACTION_STEPS}_Llama2_MoE_bz_4_new_DiT_baseline_test
echo ${SETTING}
# SETTING=freeze_none_window${FUTURE_ACTION_STEPS}_Llama2_bz_4_vla_llm_align_baseline_task3
FREEZE_VISON=true
FREEZE_LLM=False
LOAD_DIT=true
# LLM_LAYER=mix2_AdaptiveLayerDefaultDit
# LLM_LAYER=mix_avgNoneDpPooling_m2Pooling
LLM_LAYER=mix_freezeLLM_AdaptiveLayerDefaultDitsWithLearnable_test
export HF_HOME=/home/huggingface

export EMA_DECAY=${ema_decay}
# device='0,1,2,3,4,5,6,7'  
# device='4,5,6,7' 
device=$7  
# device='4'  

current_dir=$(pwd)
second_dir=$(echo "$current_dir" | cut -d"/" -f2)
export PATH=$PATH:/$second_dir/dmh/Cogact_test
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/Cogact_test  # CogAct official model
export PATH=$PATH:/$second_dir/dmh/Cogact_test/vla
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/Cogact_test/vla 



SETTING=${SETTING} COG_RES=${cog_res} RANDOM_SEED=${seed} USE_INDEX=${use_index} FOR_ROUTE=${route_in_for} TRAIN_ROUTE=False ADD_LSTM=${add_lstm} SKIP_LAYER_NUMBER=$skip_layer_number CUDA_VISIBLE_DEVICES=$device torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix rlbench \
  --vla.expected_world_size 4 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 64 \
  --vla.learning_rate 2e-5 \
  --vla.epochs 100 \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir /home/cx/4dvla_cx/dataset/mix2/${TASK} \
  --run_root_dir /home/dmh/CogACT_test/checkpoint/${TASK}_skip_layers \
  --run_id exp_cx_LLMLAYER_${LLM_LAYER}_${TASK}_${SETTING} \
  --image_aug false \
  --save_interval 600 \
  --action_dim 7 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --load_dit ${LOAD_DIT} \
  --action_model_type DiT-B \
  --is_resume False \
  --pretrained_checkpoint "/home/cx/chenhao/hub/models--CogACT--CogACT-Base/snapshots/ffc4db3bef7735ba7aa692d50b6454588a32b753/checkpoints/CogACT-Base.pt" \



#### close_box  close_laptop_lid   put_rubbish_in_bin  unplug_charger  water_plants  toilet_seat_down
