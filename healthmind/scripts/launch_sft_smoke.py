"""Launch the smoke test as a SageMaker Training job. UNTESTED: needs AWS, quota, image.
python -m scripts.launch_sft_smoke --data_s3 s3://BUCKET/smoke_data --out_s3 s3://BUCKET/smoke_out \
   --price_per_hr <verified> --budget_usd 3"""
import argparse
from infra.cost import worst_case_cost, max_minutes_for_budget
from sagemaker.pytorch import PyTorch

ROLE = "<SAGEMAKER_EXECUTION_ROLE_ARN>"
IMAGE = "<GPU_IMAGE_WITH_TORCH_TRANSFORMERS_TRL_PEFT_URI>"
ap = argparse.ArgumentParser()
ap.add_argument("--data_s3", required=True); ap.add_argument("--out_s3", required=True)
ap.add_argument("--instance", default="ml.g5.2xlarge")
ap.add_argument("--price_per_hr", type=float, required=True); ap.add_argument("--budget_usd", type=float, required=True)
ap.add_argument("--max_minutes", type=int, default=30)
a = ap.parse_args()
mins = min(a.max_minutes, max_minutes_for_budget(a.price_per_hr, a.budget_usd))
if mins < 1: raise SystemExit("budget too small")
print(f"worst-case cost ${worst_case_cost(a.price_per_hr, mins):.2f} for {mins} min")
est = PyTorch(entry_point="training/sft.py", source_dir=".", image_uri=IMAGE, role=ROLE,
              instance_count=1, instance_type=a.instance, max_run=mins * 60,
              hyperparameters={"config": "configs/smoke_sft.yaml"},
              checkpoint_s3_uri=f"{a.out_s3}/checkpoints", output_path=a.out_s3,
              environment={"MLFLOW_TRACKING_URI": "<OPTIONAL_MLFLOW_ARN_OR_URI>"})
est.fit({"train": a.data_s3})
