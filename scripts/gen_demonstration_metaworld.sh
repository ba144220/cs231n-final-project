# bash scripts/gen_demonstration_metaworld.sh basketball



cd third_party/Metaworld

task_name=${1}
num_episodes=${2:-10}

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes ${num_episodes} \
            --root_dir "../../3D-Diffusion-Policy/data/" \
