# Instructions

## Train

`python train_example_vit.py --imgs="/users/ylabrak/datasets/pneumothorax_binary_classification_task_data/" --name="pneumo_model_vit" --epochs=1`

## Predict

- Change the `config.json` to `preprocessor_config.json`
- Doesn't work with `PNG` files

`python predict.py --img="../../../samples/pneumothorax/with.jpg" --path="/users/ylabrak/Visual Transformers - ViT3/HugsVision/recipes/pneumothorax/binary_classification/out/PNEUMO_MODEL_VIT/1_2021-08-15-19-44-31/model/"`
