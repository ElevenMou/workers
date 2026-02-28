# Workers On-Call Runbook

## Scope
This runbook covers:
- Redis saturation or connection pressure
- Supabase latency/errors
- yt-dlp extraction failures
- OpenAI/API dependency failures
- Storage upload failures

## Primary Dashboards
- `GET /health/metrics`
- Redis memory + client usage
- Supabase project logs
- Worker container logs (`video-workers`, `clip-workers`, `maintenance-supervisor`)

## Alert Thresholds
- Queue age p95 above SLO for 10 minutes
- Job failure rate above 5% for 15 minutes
- Redis connection usage ratio above 0.80
- Stale recovery count increasing abnormally

## Triage Order
1. Confirm dependency health (`/ready`, Redis ping, Supabase query health).
2. Confirm queue pressure (`queues`, `queue_age_seconds`, `admissions`, `queue_rejects`).
3. Confirm worker availability (process count, restart loops, memory/CPU pressure).
4. Confirm billing safety (dedupe counters and duplicate charge checks).

## Redis Saturation
1. Check `redis.connection_usage_ratio` and `redis.used_memory_human`.
2. Scale Redis and reduce connection fan-out (`REDIS_MAX_CONNECTIONS`) if needed.
3. Temporarily increase queue rejection sensitivity by lowering queue depths.
4. Confirm recovery by watching queue age p95 and reject rates.

## Supabase Latency/Errors
1. Verify Supabase status and regional incidents.
2. Confirm retry behavior in worker logs for transient failures.
3. If persistent, reduce intake by lowering API rate limits or returning 429 earlier.
4. Monitor stale recovery and retrying job counts.

## yt-dlp Failures
1. Check worker logs for extractor failures and fallback client path.
2. Validate JS runtime availability (`node`) in worker image.
3. If source-specific, retry with degraded queue intake and notify support.

## OpenAI/API Failures
1. Check model API latency and error ratio.
2. Confirm retries are occurring and jobs are not stuck in processing.
3. If outage persists, temporarily pause new analyze requests.

## Storage Upload Failures
1. Check bucket health and object size limit errors.
2. Verify upload retry/re-encode fallback behavior in clip task logs.
3. Validate private `raw-videos` bucket accessibility for source fallback.

## Rollback Criteria
Rollback immediately if any of the following occurs:
- Duplicate processing charges observed
- Cross-node worker interference (unexpected stale-failure spikes after deploy)
- Queue age p95 doubles from baseline for > 15 minutes
