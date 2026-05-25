# Scaling Playbook (2k → 10k filings)

- Anthropic message batching
- Sharded signal cache (64 shards keyed by accession_no)
- EDGAR async semaphore at 10 req/s
- Workers KV chunking + ETag routing
- Worker bundle size budget
