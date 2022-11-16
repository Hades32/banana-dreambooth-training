import os
import json
import torch
import zipfile
import shutil
from transformers import pipeline
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

from minio import Minio
from minio.error import S3Error

s3bucket = os.environ["S3_BUCKET"]
s3client = Minio(
    os.environ["S3_ENDPOINT"],
    access_key=os.environ["S3_KEY"],
    secret_key=os.environ["S3_SECRET"],
    region=os.environ["S3_REGION"],
)

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global vae
    model = "runwayml/stable-diffusion-v1-5"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        model, 
        vae=vae, 
        torch_dtype=torch.float16, 
        revision="fp16"
    ).to("cuda")
    print("init done")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global vae

    # Parse out arguments
    data_file_id = model_inputs.get('file_id', None)

    inputS3object = 'inputs/'+data_file_id+'.zip'
    print(f"downloading {inputS3object}")
    s3client.fget_object(s3bucket, inputS3object, 'sks.zip')

    # Setup concepts_list
    concepts_list = [
        {
            "instance_prompt":      "photo of sks person",
            "class_prompt":         "photo of a person",
            "instance_data_dir":    "data/sks",
            "class_data_dir":       "data/person"
        }
    ]
    # 'class_data_dir' contains regularization images
    # 'instance_data_dir' is where training images go
    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)
    # Create concept file
    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    #Unzip training images
    train_path = concepts_list[0]["instance_data_dir"]
    with zipfile.ZipFile('sks.zip', 'r') as f:
        f.extractall('data/sks')
    
    #Call training script
    steps = 1200 # see train.sh
    train = os.system("bash train.sh")
    print("training result", train)

    print("compressing")
    #Compressed model to half size (4Gb -> 2Gb) to save space
    #compress = os.system("python convert_diffusers_to_original_stable_diffusion.py --model_path 'stable_diffusion_weights/"+str(steps)+"/' --checkpoint_path ./model.ckpt --half")
    #print(compress)
    shutil.make_archive("weights", "zip", "stable_diffusion_weights")

    weightsBucketFile = f'weights/{data_file_id}.zip'
    print(f"uploading {weightsBucketFile}")
    s3client.fput_object(
        s3bucket, weightsBucketFile, "weights.zip",
    )

    # Return the results as a dictionary
    return {'response': str(weightsBucketFile)}
