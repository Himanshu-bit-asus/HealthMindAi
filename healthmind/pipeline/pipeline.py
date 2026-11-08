"""SageMaker Pipelines definition (SKELETON, UNTESTED: needs AWS account, role,
ECR image and S3 data). Fill the placeholders, then: python pipeline/pipeline.py
Stages: process -> (sft) -> dpo -> grpo -> evaluate -> gate -> register."""
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator

ROLE = "<SAGEMAKER_EXECUTION_ROLE_ARN>"
IMAGE = "<ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/healthmind-train:<VERSION_TAG>"
BUCKET = "s3://<BUCKET>/healthmind"
session = sagemaker.session.Session()

def estimator(script, gpus_instance="ml.g5.12xlarge", spot=True):
    return Estimator(
        image_uri=IMAGE, role=ROLE, instance_count=1, instance_type=gpus_instance,
        entry_point=script, source_dir=".",
        checkpoint_s3_uri=f"{BUCKET}/checkpoints/{script}",      # auto-resume
        use_spot_instances=spot, max_run=24 * 3600, max_wait=36 * 3600 if spot else None,
        distribution={"torch_distributed": {"enabled": True}},
        sagemaker_session=session)

proc = ScriptProcessor(image_uri=IMAGE, command=["python3"], role=ROLE,
                       instance_count=1, instance_type="ml.m5.2xlarge",
                       sagemaker_session=session)

data_step = ProcessingStep(
    name="ProcessData", processor=proc, code="data/build_all.py",
    outputs=[ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
             ProcessingOutput(output_name="eval", source="/opt/ml/processing/eval")])

dpo_step = TrainingStep(
    name="DPO", estimator=estimator("training/dpo.py"),
    inputs={"train": data_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri})
# TODO: grpo_step chained on dpo_step.properties.ModelArtifacts.S3ModelArtifacts

report = PropertyFile(name="EvalReport", output_name="report", path="gate.json")
eval_step = ProcessingStep(
    name="Evaluate", processor=proc, code="evaluation/run_eval.py",   # TODO: write
    outputs=[ProcessingOutput(output_name="report", source="/opt/ml/processing/report")],
    property_files=[report])

gate = ConditionStep(
    name="Gate",
    conditions=[ConditionEquals(JsonGet(step_name=eval_step.name, property_file=report,
                                        json_path="passed"), True)],
    if_steps=[],     # TODO: RegisterModel, then manual approval
    else_steps=[])

pipeline = Pipeline(name="healthmind", steps=[data_step, dpo_step, eval_step, gate],
                    sagemaker_session=session)
if __name__ == "__main__":
    pipeline.upsert(role_arn=ROLE); pipeline.start()
