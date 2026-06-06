# uv run predict.py --model model
import os
import argparse

models = ["EnGAN", "Colie"]
model = "EnGAN"


if model == "EnGAN":
    os.system("uv run EnGAN/scripts/script.py --predict")
    # predict all datasets using EnGAN
    # 调用EnGAN的predict(dataroot==datasets) =》