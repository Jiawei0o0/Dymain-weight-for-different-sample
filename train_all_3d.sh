
# all
python ./code/train_mine_all.py --dataset_name LA --model mine3d_v1 --labelnum 4 --gpu 2 --temperature 0.1 &&\
python ./code/train_mine_all.py --dataset_name LA --model mine3d_v1 --labelnum 8 --gpu 2 --temperature 0.1 
python ./code/test_3d.py --dataset_name LA --model mine3d_v1 --exp MCNet --labelnum 4 --gpu 1 &&\
python ./code/test_3d.py --dataset_name LA --model mine3d_v1 --exp MCNet --labelnum 8 --gpu 1 

# Pancreas_CT
# python ./code/train_mine_all.py --dataset_name Pancreas_CT --model mine3d_v1 --labelnum 6 --gpu 0 --temperature 0.1 &&\
# python ./code/train_mine_all.py --dataset_name Pancreas_CT --model mine3d_v1 --labelnum 12 --gpu 0 --temperature 0.1 

# python ./code/test_3d.py --dataset_name Pancreas_CT --model mine3d_v1 --exp MCNet --labelnum 12 --gpu 2 && \
# python ./code/test_3d.py --dataset_name Pancreas_CT --model mine3d_v1 --exp MCNet --labelnum 6 --gpu 2
