# HealthMind AI training scaffold

## Status
| Component | Status |
|---|---|
| training/adapter_chain.py | Implemented, unit-tested locally |
| evaluation/gate.py | Implemented, unit-tested locally (thresholds are placeholders, set with clinicians) |
| tools/diet_calculator.py | Implemented, unit-tested locally |
| rewards/medical_rewards.py | Implemented, unit-tested (structure, evidence, escalation, uncertainty only; NOT clinical correctness) |
| data/rlaif.py | Implemented, unit-tested (judge validation, position-swap check, review routing, dedupe cache) |
| data/sources.py, data/ingest.py, data/contamination.py | Implemented, unit-tested (license gate, de-identification first pass, chunking, exact/near dedupe, eval-overlap check) |
| evaluation/baseline.py | Implemented, unit-tested with fake models; NOT yet run on II-Medical-8B |
| data/retrieval.py, data/teacher_gen.py | Implemented, unit-tested with a fake teacher (cited-answer generation, rejection checks, review queue, role-checked review, approved-only SFT export). Real teacher endpoint NOT connected |
| data/teacher_client.py, scripts/run_teacher_gen.py | Implemented; HTTP behaviour (retry, cache, budget, privacy guard) tested against a local fake server. NOT yet tried against a real teacher model |
| review/service.py | Implemented, unit-tested (role + verification checks, leases, dual review for safety-critical, dispute/adjudication, audit log, approved-only export) |
| review/api.py, infra/review_schema.sql | Written, NOT run (FastAPI/Postgres not available in my sandbox; syntax-checked only). get_reviewer is a placeholder you must wire to your auth |
| training/sft_data.py | Implemented, unit-tested (approved-provenance only, eval-overlap removal, dedupe, min-size guard, seeded split, content-hash dataset version, sft_needed check) |
| training/sft.py | Skeleton, NOT run (needs GPU + pinned TRL; verify assistant-only loss in your TRL version) |
| evaluation/baseline_job.py, scripts/launch_baseline.py | Written, NOT run (needs GPU + AWS). Runs data-parallel across all GPUs (shard/merge logic unit-tested). Records GPU name, peak VRAM per GPU, seconds. Reasoning (<think>) blocks are stripped before scoring (tested) |
| training/loading.py | QLoRA plan logic unit-tested; load_model NOT run (needs GPU + bitsandbytes). Adapter manifests now record quantization and a mismatched parent is refused (tested) |
| scripts/make_smoke_data.py, training/monitor.py, scripts/launch_sft_smoke.py, docs/SMOKE_TEST.md | Smoke-test data generator, loss/NaN report, release guard: tested. Launcher + VRAM callback NOT run (need AWS/GPU). Smoke adapters are blocked from reuse and release |
| training/dpo.py | Skeleton, NOT run (needs GPU + pinned TRL) |
| pipeline/pipeline.py | Skeleton, NOT run (needs AWS) |
| SFT, GRPO trainer, judge client, data build, eval scripts, RAG, cost reports | Not written yet |

No benchmark or cost numbers exist yet. Measure a baseline on a small g5 run first.

## Run the tests
    pip install -r requirements.txt
    python -m pytest tests/unit -q

## Run teacher generation
    export JUDGE_BASE_URL=https://your-private-endpoint/v1 JUDGE_MODEL=your-model JUDGE_API_KEY=...
    python -m scripts.run_teacher_gen --chunks corpus/chunks.jsonl --questions questions.jsonl --out out/ --max_calls 50
Start with ~50 questions and read out/report.json before scaling.

## Compute plan
Budget: $114 credit. ml.g6.48xlarge is ~US$16.7/hr on-demand per third-party trackers (verify on aws.amazon.com/sagemaker/pricing) = under 7 hours of credit, so use it only if at all for short final runs. Develop on a single-GPU instance; launch_baseline.py now REQUIRES --price_per_hr and --budget_usd and caps runtime to the budget.
Earlier target (large): ml.g6.48xlarge = 8x NVIDIA L4 24GB (bf16 supported; low memory bandwidth, PCIe only). Precision/attention auto-selected by training/hw.py (selection logic tested). 8B bf16 weights ~16GB leave little headroom on a 24GB GPU: training starts at batch 1 + accumulation 16; measure peak VRAM, and QLoRA is now on by default in configs/sft.yaml and dpo.yaml (qlora: true); its real memory use is UNMEASURED, check peak VRAM. Run launch_baseline.py --smoke first to measure real speed; any time estimate before that is a guess.
ml.t3.medium (no GPU) = control machine only: tests, ingestion, review service, launching jobs.
GPU work (baseline, SFT, DPO, GRPO) runs as SageMaker jobs. Check Service Quotas for GPU instance limits first.

## Next steps
1. Write data/build_all.py and evaluation/run_eval.py.
2. Smoke-test dpo.py on one GPU, then pin versions.
3. Add grpo.py + rewards, then finish pipeline.py.
