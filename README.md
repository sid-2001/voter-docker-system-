## Parallel OCR workers with crash recovery

### Run
```bash
docker compose up --build
```

This starts:
- `mongodb`
- `worker1`, `worker2`, `worker3`

Each worker claims one PDF job at a time from MongoDB (`srs_pdf_jobs`) using atomic `find_one_and_update`, so workers do not process the same PDF.

### Crash recovery
- Each processing worker writes a `heartbeat_at` timestamp.
- If a worker crashes and its heartbeat is stale (`STALE_SECONDS`), another worker will reclaim and continue that PDF job.

### Logs
- Terminal logs: `docker compose up`
- File logs: `./logs/worker-<id>.log`
- Each log line includes global counters:
  - total PDFs done/pending/processing/failed
  - total records in main collection

### Timing per PDF
Per PDF elapsed time is stored in `srs_pdf_jobs.metrics.elapsed_seconds` and logged in terminal/file.
