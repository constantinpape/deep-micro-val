# TODO adapt env names

# segment with baselines
echo "Baselines ..."
# source activate torch-em-test
# python shila.py $1 -s 1

# segment with stardist
echo "Stardist ..."
source activate stardist-cpu
python shila.py $1 -s 1

# segment with cellpose
echo "Cellpose ..."
source activate cellpose-cpu
python shila.py $1 -s 1
