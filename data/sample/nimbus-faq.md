# Nimbus Analytics Platform — Frequently Asked Questions

> Synthetic documentation for DocRAG testing. Not a real product.

### How often does the Nimbus Collector send events?

The Collector flushes its buffer every 5 seconds, or when the batch reaches 500
events, whichever comes first.

### How long does Nimbus retain raw event data?

Raw events are retained for 90 days. Daily aggregates are retained for 5 years.

### What is the maximum time range for a single query?

A single Query API request can span at most 13 months.

### Which authentication methods are supported?

Nimbus supports API keys for programmatic access and SAML 2.0 single sign-on for
the dashboard. API keys can be scoped to read-only or read-write.

### How do I rotate an API key?

Open Settings → API Keys and click "Rotate". The old key remains valid for a
24-hour grace period to allow a smooth migration.

### Is there a free tier?

Yes. The Starter tier is free for up to 1 million events per month and includes
community support.

### How do I request a data export?

Enterprise customers can request a full export via the Query API's `/export`
endpoint, which returns gzipped JSON. Exports are capped at 10 GB per request.
