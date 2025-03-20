TASK=10tasks_selected_keyframe_state
FUTURE_ACTION_STEPS=15
skip_layer_number=$1
mse_weight=$2
balance_weight=$3
KD_weight=$4
add_lstm=False
route_in_for=False
use_index=False 
seed=$5
ema_decay=$6
# SETTING=freeze_vit_window${FUTURE_ACTION_STEPS}
SETTING=route_in_for_${route_in_for}_balance_${balance_weight}_KD_${KD_weight}_ema_${ema_decay}_seed_${seed}_skip_layer_${skip_layer_number}_mse_${mse_weight}_freeze_none_window${FUTURE_ACTION_STEPS}
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


# device='0,1,2,3,4,5,6,7' 
# device='4,5,6,7' 
device=$7  
# device='4' 

export PATH=$PATH:/path/to/MoLE_VLA
export PYTHONPATH=$PYTHONPATH:/path/to/MoLE_VLA
export PATH=$PATH:/path/to/MoLE_VLA/vla
export PYTHONPATH=$PYTHONPATH:/path/to/MoLE_VLA/vla 

export EMA_DECAY=${ema_decay}
export MSE=${mse_weight}
export BALANCE=${balance_weight}
export KD=${KD_weight}

SETTING=${SETTING} COG_RES=${cog_res} RANDOM_SEED=${seed} USE_INDEX=${use_index} FOR_ROUTE=${route_in_for} TRAIN_ROUTE=False ADD_LSTM=${add_lstm} SKIP_LAYER_NUMBER=$skip_layer_number CUDA_VISIBLE_DEVICES=$device torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix rlbench \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 512 \
  --vla.per_device_batch_size 64 \
  --vla.learning_rate 2e-5 \
  --vla.epochs 100 \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir ./dataset/${TASK} \
  --run_root_dir /path/to/save_dir \
  --run_id exp_cx_LLMLAYER_${LLM_LAYER}_${TASK}_${SETTING} \
  --image_aug false \
  --save_interval 800 \
  --action_dim 7 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --load_dit ${LOAD_DIT} \
  --action_model_type DiT-B \
  --is_resume False \
  --pretrained_checkpoint "/path/to/pretrained_cogact_base" \

#### close_box  close_laptop_lid   put_rubbish_in_bin  unplug_charger  water_plants  toilet_seat_down
