# Run the main generation program with optimus

cd ../code
export code_path=$(pwd)
echo $code_path
export PYTHONPATH="${PYTHONPATH}:$code_path"

export TRAIN_FILE=../data/worldtree/train.txt
export TEST_FILE=../data/worldtree/test.txt

cd ../

python code/big_ae/run_latent_generation.py \
--checkpoint_dir=output/worldtree_ckpt \
--output_dir=output/worldtree_ckpt \
--decoder_model_type=gpt2 \
--decoder_model_name_or_path=gpt2 \
--train_data_file=$TRAIN_FILE \
--eval_data_file=$TEST_FILE \
--per_gpu_eval_batch_size=1 \
--gloabl_step_eval 3000 \
--block_size 100 \
--max_seq_length 100 \
--interact_with_user_input \
--play_mode interpolation \
--num_interpolation_steps=10


