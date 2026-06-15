# Getting Started with Nimbus — Onboarding Guide

> Synthetic documentation for DocRAG testing. Not a real product.

## Step 1 — Create a project

Sign in to the Nimbus console and click "New Project". Each project has its own
isolated warehouse and API keys. You can create up to 25 projects per account on
the Growth tier.

## Step 2 — Install the Collector

Install the Collector agent with the package manager for your platform. The agent
requires outbound HTTPS access to `ingest.nimbus.example` on port 443.

## Step 3 — Send your first event

Use a read-write API key to POST a JSON event to the ingest endpoint. A valid
event must include a `name`, a `timestamp` in ISO 8601 format, and an optional
`properties` object.

## Step 4 — Build a dashboard

Open the Dashboards tab and add a chart. Charts can group by any event property
and support line, bar, and stacked-area visualizations.

## Troubleshooting

- **Events not appearing?** Confirm the API key is read-write and that the event
  timestamp is within the last 13 months.
- **Collector failing to connect?** Verify outbound access to port 443 and that
  your system clock is accurate to within 5 minutes.
- **Hitting rate limits?** The ingest endpoint allows 10,000 requests per minute
  per project on the Growth tier; upgrade to Enterprise for higher limits.
