import os
import pathlib
import time
import json
import torch
import zipfile
import shutil
from transformers import pipeline
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

from minio import Minio
from minio.error import S3Error

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    # global vae
    # model = "runwayml/stable-diffusion-v1-5"
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    
    # global pipe
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     model, 
    #     vae=vae, 
    #     torch_dtype=torch.float16, 
    #     revision="fp16"
    # ).to("cuda")
    print("init done")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    #global model
    #global vae

    # Parse out arguments
    s3bucket = model_inputs.get("S3_BUCKET", os.getenv("S3_BUCKET"))
    s3client = Minio(
        model_inputs.get("S3_ENDPOINT", os.getenv("S3_ENDPOINT")),
        access_key=model_inputs.get("S3_KEY", os.getenv("S3_KEY")),
        secret_key=model_inputs.get("S3_SECRET", os.getenv("S3_SECRET")),
        region=model_inputs.get("S3_REGION", os.getenv("S3_REGION"))
    )
    s3_input_path = model_inputs.get('input_path', None)
    s3_output_path = model_inputs.get('output_path', None)

    if s3_input_path is None:
        return {error: "no input"}

    # Setup concepts_list
    concepts_list = [
        {
            "instance_prompt":      "photo of sks person",
            "class_prompt":         "photo of a person",
            "instance_data_dir":    "data/sks",
            "class_data_dir":       "data/person"
        }
    ]
    train_path = concepts_list[0]["instance_data_dir"]

    print(f"downloading {s3_input_path}")
    objects = s3client.list_objects(s3bucket, prefix=s3_input_path)
    for obj in objects:
        s3client.fget_object(s3bucket, obj.object_name, pathlib.posixpath.join(train_path, pathlib.posixpath.basename(obj.object_name)))

    # 'class_data_dir' contains regularization images
    # 'instance_data_dir' is where training images go
    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)
    # Create concept file
    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)
    
    #Call training script
    steps = 1200 # see train.sh
    train = os.system("bash train.sh")
    print("training result", train)

    print("compressing")
    #Compressed model to half size (4Gb -> 2Gb) to save space
    #compress = os.system("python convert_diffusers_to_original_stable_diffusion.py --model_path 'stable_diffusion_weights/"+str(steps)+"/' --checkpoint_path ./model.ckpt --half")
    #print(compress)
    shutil.make_archive("weights", "zip", f"stable_diffusion_weights/{steps}")

    uploadStart = time.monotonic_ns()
    print(f"uploading {s3_output_path}")
    s3client.fput_object(
        s3bucket, s3_output_path, "weights.zip", part_size=100*1024*1024,
    )
    print(f"finished uploading in {(time.monotonic_ns() - uploadStart)/1_000_000_000}s")

    # Return the results as a dictionary
    return {'response': str(s3_output_path)}
