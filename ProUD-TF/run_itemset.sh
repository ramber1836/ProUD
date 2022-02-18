for c in {1..5}
do
    python itemset.py --dir ../nyc/ --cuda_devices 4 --patience 5 --action_feat_size 1
done

