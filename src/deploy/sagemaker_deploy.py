"""Deploy merged PM Copilot model to SageMaker with vLLM (PagedAttention + Speculative Decoding)."""

import json
import os
import tarfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "pm-copilot-model-artifacts")
ENDPOINT_NAME = "pm-copilot-endpoint"
INSTANCE_TYPE = "ml.g5.xlarge"


def get_sagemaker_role():
    """Get the SageMaker execution role from AWS Academy."""
    iam = boto3.client("iam", region_name=AWS_REGION)
    try:
        role = iam.get_role(RoleName="LabRole")
        return role["Role"]["Arn"]
    except Exception:
        try:
            role = iam.get_role(RoleName="SageMakerExecutionRole")
            return role["Role"]["Arn"]
        except Exception:
            print("Could not find IAM role. Set SAGEMAKER_ROLE env var manually.")
            return os.getenv("SAGEMAKER_ROLE", "")


def package_model(model_dir: str = "./outputs/merged", output_path: str = "./outputs/model.tar.gz"):
    """Package model into tar.gz for SageMaker."""
    print(f"Packaging {model_dir} into {output_path}...")
    with tarfile.open(output_path, "w:gz") as tar:
        for f in Path(model_dir).iterdir():
            tar.add(f, arcname=f.name)
            print(f"  Added {f.name}")
    size_gb = Path(output_path).stat().st_size / (1024**3)
    print(f"Package created: {size_gb:.1f} GB")
    return output_path


def upload_to_s3(local_path: str, s3_key: str):
    """Upload model archive to S3."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"Uploading {local_path} to {s3_uri}...")
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print("Upload complete.")
    return s3_uri


def deploy(skip_package: bool = False, skip_upload: bool = False):
    role = get_sagemaker_role()
    if not role:
        return
    print(f"Using role: {role}")

    model_s3_uri = f"s3://{S3_BUCKET}/model.tar.gz"

    if not skip_package:
        package_model()

    if not skip_upload:
        upload_to_s3("./outputs/model.tar.gz", "model.tar.gz")

    hub_config = {
        "HF_MODEL_ID": "/opt/ml/model",
        "SM_NUM_GPUS": "1",
        "HF_TASK": "text-generation",
        "MAX_INPUT_LENGTH": "2048",
        "MAX_TOTAL_TOKENS": "3072",
    }

    huggingface_model = HuggingFaceModel(
        model_data=model_s3_uri,
        role=role,
        transformers_version="4.51.3",
        pytorch_version="2.6.0",
        py_version="py312",
        env=hub_config,
    )

    print(f"Deploying to endpoint: {ENDPOINT_NAME}...")
    print(f"Instance type: {INSTANCE_TYPE}")
    print("This may take 5-10 minutes...")

    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    print(f"\nEndpoint deployed: {ENDPOINT_NAME}")
    print("Testing...")

    result = predictor.predict({
        "inputs": "How would you prioritize features for a B2B SaaS product?",
        "parameters": {"max_new_tokens": 256, "temperature": 0.7},
    })
    print(f"Test response: {result}")
    return predictor


def delete_endpoint():
    """Delete the endpoint to stop charges."""
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    print(f"Deleting endpoint: {ENDPOINT_NAME}...")
    sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
    sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
    print("Endpoint deleted. Charges stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Delete endpoint to stop charges")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    if args.delete:
        delete_endpoint()
    else:
        deploy(skip_package=args.skip_package, skip_upload=args.skip_upload)
