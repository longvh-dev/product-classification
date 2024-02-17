set -e

pythonpath='python'


${pythonpath} train.py \
            --data_path data/data.txt \
            --test_percent 0.2 \
            --random_state 42 \
            --model_path save/ \
            --is_preprocess False \
            --is_train True \
            --is_evaluate True