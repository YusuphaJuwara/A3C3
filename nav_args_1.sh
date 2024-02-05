export PYTHONPATH=$(pwd)

echo "starting"
for trial in 0
do
    python.exe main.py --test=False --load_model=False --num_workers=5 --num_agents=2 --comm_size=20 --critic=0 --max_epis=150000 --save_dir="models" --comm_gaussian_noise=0.0 --comm_delivery_failure_chance=0.0 --comm_jumble_chance=0.0 --param_search==40,relu,80,40
done