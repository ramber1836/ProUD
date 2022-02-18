for dim in 16 32 48 64 80 96 112
do
    for c in {1..10}
    do
        python decision.py --dir ../nyc/ --dim $dim --cuda_devices 7 --action_feat_size 1
    done
done
