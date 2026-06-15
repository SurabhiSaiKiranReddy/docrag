# Nimbus Analytics Platform — Product Overview

> This is **synthetic documentation** created for testing DocRAG. It does not
> describe any real product, company, or proprietary system.

## What is Nimbus?

Nimbus is a fictional cloud-native analytics platform that ingests event data,
stores it in a columnar warehouse, and exposes dashboards and an API for
querying aggregated metrics. It was first released in March 2021.

## Core Components

- **Nimbus Collector** — a lightweight agent that batches events and ships them
  over HTTPS every 5 seconds. The default batch size is 500 events.
- **Nimbus Warehouse** — a columnar store that retains raw events for 90 days
  and rolled-up daily aggregates for 5 years.
- **Nimbus Query API** — a REST API that supports filtering, grouping, and
  time-bucketed aggregation. The maximum query range is 13 months.
- **Nimbus Dashboards** — a web UI for building charts without writing code.

## Service Level Agreement (SLA)

The Nimbus Enterprise tier guarantees 99.95% monthly uptime. If uptime falls
below 99.95%, customers receive a 10% service credit; below 99.0%, a 30%
service credit is issued automatically.

## Pricing Tiers

| Tier       | Monthly events | Price (USD/month) | Support      |
|------------|----------------|-------------------|--------------|
| Starter    | up to 1M       | $0                | Community    |
| Growth     | up to 50M      | $499              | Email, 24h   |
| Enterprise | unlimited      | Custom            | 24/7 phone   |

## Supported Regions

Nimbus runs in four regions: us-east-1, us-west-2, eu-west-1, and ap-south-1.
Data residency is guaranteed within the selected region.

## Security

All data is encrypted in transit with TLS 1.3 and at rest with AES-256. Nimbus
is SOC 2 Type II certified and supports single sign-on via SAML 2.0.
