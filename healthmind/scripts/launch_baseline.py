"""Run from a cheap notebook (e.g. ml.t3.medium). Starts a GPU Processing job, then the
instance can be stopped. UNTESTED: needs AWS, a GPU quota and a container image.
Fill ROLE, IMAGE, S3 paths. Check Service Quotas: GPU processing instances are often at 0."""
import argparse
from infra.cost import worst_case_cost, max_minutes_for_budget
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

ROLE = "<SAGEMAKER_EXECUTION_ROLE_ARN>"
IMAGE = "<GPU_PYTORCH_IMAGE_WITH_TRANSFORMERS_URI>"   # build from infra/Dockerfile and push to ECR

ap = argparse.ArgumentParser()
ap.add_argument("--eval_s3", required=True); ap.add_argument("--out_s3", required=True)
ap.add_argument("--instance", default="ml.g5.2xlarge")   # 1x A10G 24GB; verify availability + quota in your region
ap.add_argument("--gpus", type=int, default=1)   # must match the instance GPU count
ap.add_argument("--price_per_hr", type=float, required=True, help="verified USD/hr for this instance+region")
ap.add_argument("--budget_usd", type=float, required=True, help="hard cap for THIS job")
ap.add_argument("--max_minutes", type=int, default=45)   # raise after a smoke test shows real speed
ap.add_argument("--smoke", action="store_true", help="score only 32 items to measure speed first")    # hard stop so a hung job can't keep billing
a = ap.parse_args()
cap = max_minutes_for_budget(a.price_per_hr, a.budget_usd)
if cap < 1: raise SystemExit("budget too small for even one minute at this price")
a.max_minutes = min(a.max_minutes, cap)          # runtime limit can never exceed the budget
print(f"worst-case cost: ${worst_case_cost(a.price_per_hr, a.max_minutes):.2f} "
      f"for {a.max_minutes} min (budget ${a.budget_usd:.2f})")

p = ScriptProcessor(image_uri=IMAGE, command=["torchrun", f"--nproc_per_node={a.gpus}"], role=ROLE, instance_count=1,
                    instance_type=a.instance, volume_size_in_gb=100,
                    max_runtime_in_seconds=a.max_minutes * 60,
                    env={"BASE_MODEL": "Intelligent-Internet/II-Medical-8B", "MODEL_REVISION": "<COMMIT_HASH>",
                         "MAX_ITEMS": "32" if a.smoke else "0"})
p.run(code="evaluation/baseline_job.py", source_dir=".",
      inputs=[ProcessingInput(source=a.eval_s3, destination="/opt/ml/processing/input")],
      outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination=a.out_s3)])
