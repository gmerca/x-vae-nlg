# Run script for run_lm_vae_training.py optimus fine-tuning on worldtree dataset using wikipedia checkpoint

cd ../code
export code_path=$(pwd)
echo $code_path
export PYTHONPATH="${PYTHONPATH}:$code_path"

export TRAIN_FILE=../data/worldtree/train.txt
export TEST_FILE=../data/worldtree/test.txt

cd ../

python code/big_ae/run_lm_vae_training.py \
--output_dir=output/wt_wiki_finetuning \
--train_data_file=$TRAIN_FILE \
--eval_data_file=$TEST_FILE \
--dataset worldtree \
--checkpoint_dir=output/wikipedia-ckpt \
--encoder_model_type=bert  \
--encoder_model_name_or_path=bert-base-cased  \
--decoder_model_type=gpt2  \
--decoder_model_name_or_path=gpt2  \
--beta 1.0  \
--ratio_zero 0.5  \
--ratio_increase 0.25  \
--do_train  \
--do_eval  \
--fb_mode 1  \
--dim_target_kl 0.5 \
--num_train_epochs 1.0 \
--save_steps 1000  \
--logging_steps 1000  \
--overwrite_output_dir \
--block_size 100  \
--use_pretrained_model  \
--use_pretrained_vae \
--gloabl_step_eval 508523 \
--per_gpu_train_batch_size=1 \

