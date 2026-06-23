# Customer Database Seeder

Initializes and continuously maintains a small, active demo dataset in a PostgreSQL
customer database. Designed to run as a sidecar alongside the database container.

## Behavior

- **First run (empty database):** creates the schema (`customers`, `products`,
  `purchases`), seeds reference data, and backfills ~90 days of purchase history.
- **Every run:** enters an hourly loop that inserts a fresh batch of purchases dated
  today and prunes purchases older than the retention window.
- **Idempotent:** on restart with data already present, it skips seeding and goes
  straight to the loop (reference data is seeded only once, so unique constraints are
  never violated and purchases do not accumulate unboundedly).

## Configuration

### Required Environment Variables

- `PGHOST`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`: PostgreSQL connection. When run as a
  sidecar, `PGHOST=localhost`.

### Optional Environment Variables

- `PGPORT`: `5432` by default.
- `SEED_CUSTOMERS`: `50` by default.
- `SEED_PRODUCTS`: `20` by default.
- `BACKFILL_DAYS`: `90` by default. Window of backfilled purchase history.
- `BACKFILL_PURCHASES`: `400` by default. Number of purchases backfilled on first run.
- `RETENTION_DAYS`: `90` by default. Purchases older than this are pruned each tick.
- `HOURLY_PURCHASES`: `5` by default. Purchases inserted per tick.
- `INSERT_INTERVAL_SECONDS`: `3600` by default. Loop interval.

## Build Locally

```bash
docker build -t customer-database-seeder .
```
