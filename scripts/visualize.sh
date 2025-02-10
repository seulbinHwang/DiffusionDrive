export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
	--result-path work_dirs/diffusiondrive_small_stage2/results.pkl