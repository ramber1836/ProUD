for reg_weight in 0 0.003 0.01 0.03 0.1 0.3 1
do
    for c in {1..10}
    do
        python decision.py --dir ../nyc/ --reg_weight $reg_weight --cuda_devices 6 --action_feat_size 1
    done
done

# python decision.py --dir ../nyc/ --reg_weight 0.01 --cuda_devices 6