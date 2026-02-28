# Chaos Scenarios (Staging)

Run these scenarios before production rollout:

1. Redis restart during active processing
- Start active analyze and generate jobs.
- Restart Redis once.
- Validate jobs move to retrying/failed/completed without indefinite processing state.
- Validate stale recovery counters increase only for truly orphaned jobs.

2. Supabase transient failures
- Inject temporary network loss or 5xx responses.
- Confirm retries for transient failures and no duplicate billing charges.

3. OpenAI timeout burst
- Inject repeated timeout failures for analyzer calls.
- Confirm queue does not collapse and retrying jobs are visible.

4. Worker pod/container eviction
- Evict one worker instance while jobs are running.
- Confirm maintenance stale sweep and admission release prevent stuck capacity.

5. Cross-node raw source fallback
- Analyze on node A, generate on node B.
- Confirm source resolution succeeds via canonical `raw-videos` object.
