# Instructions

## Train

`python train_example_deit.py --imgs="/users/ylabrak/datasets/kvasir-dataset-v2/" --epochs=20`

## Predict

- Change the `config.json` to `preprocessor_config.json`
- Doesn't work with `PNG` files

`python predict_deit.py --img="../../../samples/kvasir_v2/dyed-lifted-polyps.jpg" --path="./out/KVASIR_V2_MODEL_DEIT/20_2021-08-20-01-46-44/model/"`
