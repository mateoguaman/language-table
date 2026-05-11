export GOOGLE_API_KEY="AIzaSyAIyKWeVkmuRLxIjuuLJglrts0TVv4eSco"
VLA_CKPT="/home/memmelma/Projects/metarl/language-table/bc_resnet_sim_checkpoint_955000"


# arrange the blue and green blocks in a horizontal line
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 ./ltvenv/bin/python -m language_table.lamer.server_main   --host 127.0.0.1   --port 50051   --num_envs 1   --group_n 1   --block_mode BLOCK_8   --max_inner_steps 20   --num_attempts 1   --max_turns 20  --reward_type custom   --reward_kwargs '{"provider":"TetrisTaskProvider","shapes":["I","O","T","L"],"seed":0,"instruction_template":"arrange the blue and green blocks in a horizontal line", "dummy_zero_reward":true}'   --split train   --preprocess_mode jax_gpu   --vla_checkpoint "$VLA_CKPT"   --include_rgb

# arrange the blocks into the tetris/tetromino shape: {letter} | T
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 ./ltvenv/bin/python -m language_table.lamer.server_main   --host 127.0.0.1   --port 50051   --num_envs 1   --group_n 1   --block_mode BLOCK_4   --max_inner_steps 20   --num_attempts 1   --max_turns 20  --reward_type custom   --reward_kwargs '{"provider":"TetrisTaskProvider","shapes":["T"],"seed":0,"instruction_template":"arrange the blocks into the tetris/tetromino shape: {letter}", "dummy_zero_reward":true}'   --split train   --preprocess_mode jax_gpu   --vla_checkpoint "$VLA_CKPT"   --include_rgb