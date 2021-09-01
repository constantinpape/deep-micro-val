# TODO adapt env names

# segment with baselines
echo "Baselines ..."
source activate torch-em-test
python dsb.py $1 -s 1

# segment with stardist
echo "Stardist ..."
source activate stardist-cpu
python dsb.py $1 -s 1

# segment with cellpose
echo "Cellpose ..."
source activate cellpose-cpu
python dsb.py $1 -s 1

# run evaluation
echo "Evaluation ..."
source activate torch-em-test
python dsb.py $1 -e 1
