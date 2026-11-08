# Smoke test: verify the training code before spending real credit

Uses synthetic arithmetic data and a 0.6B model. Contains NO medical content. Produces an
adapter that the adapter chain refuses to continue from or release.

1. python -m scripts.make_smoke_data --out smoke_data
2. aws s3 cp smoke_data s3://BUCKET/smoke_data --recursive
3. Look up the price of your instance in your region; then:
   python -m scripts.launch_sft_smoke --data_s3 s3://BUCKET/smoke_data --out_s3 s3://BUCKET/smoke_out --price_per_hr <PRICE> --budget_usd 3
4. Read train_report.json in the output (S3 model.tar.gz).

PASS only if ALL hold:
- job finishes; train_report.json has "ok": true (no NaN/inf, loss decreased)
- peak_vram_gb is logged (this is your first real memory measurement)
- checkpoints appear under smoke_out/checkpoints
- resume works: run the same command again with the same checkpoint path (or use managed spot);
  the log should say it resumed from a checkpoint instead of starting at step 0

If it fails, fix before running anything on II-Medical-8B. A pass proves the code path runs;
it says nothing about medical quality.
