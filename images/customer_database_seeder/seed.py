#!/usr/bin/env python3
"""Customer database seeder.

Initializes and continuously maintains a small, active demo dataset in PostgreSQL.

On first run (empty database) it creates the schema, seeds reference data (customers and
products), and backfills ~90 days of purchase history. On every run it then enters an
hourly loop that inserts a fresh batch of purchases dated ~now and prunes purchases older
than the retention window. The init step is idempotent: on restart (data already present)
it skips seeding and goes straight to the loop.
"""

import logging
import os
import random
import time
from datetime import date, timedelta

import pg8000.dbapi
from faker import Faker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("customer_database_seeder")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

PGHOST = os.environ.get("PGHOST", "localhost")
PGPORT = int(os.environ.get("PGPORT", "5432"))
PGDATABASE = os.environ.get("PGDATABASE", "customers")
PGUSER = os.environ.get("PGUSER", "customers")
PGPASSWORD = os.environ.get("PGPASSWORD", "")

SEED_CUSTOMERS = int(os.environ.get("SEED_CUSTOMERS", "50"))
SEED_PRODUCTS = int(os.environ.get("SEED_PRODUCTS", "20"))
BACKFILL_DAYS = int(os.environ.get("BACKFILL_DAYS", "90"))
BACKFILL_PURCHASES = int(os.environ.get("BACKFILL_PURCHASES", "400"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "90"))
HOURLY_PURCHASES = int(os.environ.get("HOURLY_PURCHASES", "5"))
INSERT_INTERVAL_SECONDS = int(os.environ.get("INSERT_INTERVAL_SECONDS", "3600"))
STARTUP_MAX_RETRIES = int(os.environ.get("STARTUP_MAX_RETRIES", "30"))
STARTUP_RETRY_SECONDS = int(os.environ.get("STARTUP_RETRY_SECONDS", "5"))


# --------------------------------------------------------------------------------------
# Schema (PostgreSQL)
# --------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    birth_date      DATE NOT NULL,
    street_address  TEXT NOT NULL,
    city            TEXT NOT NULL,
    state           TEXT NOT NULL,
    zip_code        TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    phone_number    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sku             TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL,
    cost            NUMERIC(10, 2) NOT NULL,
    in_stock_qty    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS purchases (
    id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    product_id      INTEGER NOT NULL REFERENCES products(id),
    qty_purchased   INTEGER NOT NULL,
    purchase_date   DATE NOT NULL
);
"""


# --------------------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------------------


def connect() -> pg8000.dbapi.Connection:
    return pg8000.dbapi.connect(
        host=PGHOST,
        port=PGPORT,
        database=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
    )


def wait_for_database() -> pg8000.dbapi.Connection:
    """Wait for PostgreSQL to accept connections (the postgres sidecar may start later)."""
    for attempt in range(1, STARTUP_MAX_RETRIES + 1):
        try:
            conn = connect()
            logger.info("Connected to PostgreSQL at %s:%s", PGHOST, PGPORT)
            return conn
        except Exception as exc:  # noqa: BLE001 - retry on any connection error
            logger.info(
                "Waiting for PostgreSQL (attempt %d/%d): %s",
                attempt,
                STARTUP_MAX_RETRIES,
                exc,
            )
            time.sleep(STARTUP_RETRY_SECONDS)
    raise RuntimeError("PostgreSQL did not become available in time")


def get_id_ranges(conn: pg8000.dbapi.Connection) -> tuple[int, int]:
    """Return (max customer id, max product id) for random purchase generation."""
    cursor = conn.cursor()
    cursor.execute("SELECT max(id) FROM customers")
    max_customer = cursor.fetchone()[0]
    cursor.execute("SELECT max(id) FROM products")
    max_product = cursor.fetchone()[0]
    return max_customer, max_product


# --------------------------------------------------------------------------------------
# Fake data generation
# --------------------------------------------------------------------------------------


def make_customer(fake: Faker) -> tuple:
    return (
        fake.first_name(),
        fake.last_name(),
        fake.date_of_birth(minimum_age=18, maximum_age=85),
        fake.street_address(),
        fake.city(),
        fake.state_abbr(),
        fake.zipcode(),
        fake.unique.email(),
        fake.phone_number(),
    )


def make_product(fake: Faker) -> tuple:
    return (
        fake.bothify(text="SKU-####-???").upper(),
        fake.catch_phrase(),
        fake.sentence(nb_words=12),
        round(random.uniform(4.99, 299.99), 2),
        random.randint(0, 500),
    )


def make_purchase(max_customer: int, max_product: int, purchase_date: date) -> tuple:
    return (
        random.randint(1, max_customer),
        random.randint(1, max_product),
        random.randint(1, 5),
        purchase_date,
    )


# --------------------------------------------------------------------------------------
# Seed + backfill (first run only)
# --------------------------------------------------------------------------------------


def initialize(conn: pg8000.dbapi.Connection) -> None:
    """Create the schema and, if empty, seed reference data and backfill purchases."""
    cursor = conn.cursor()
    cursor.execute(SCHEMA)
    conn.commit()

    cursor.execute("SELECT count(*) FROM customers")
    if cursor.fetchone()[0] > 0:
        logger.info("Database already seeded; skipping initial seed and backfill")
        return

    # Deterministic reference data for a stable demo baseline.
    fake = Faker()
    Faker.seed(42)
    random.seed(42)

    logger.info("Seeding %d customers", SEED_CUSTOMERS)
    customers = [make_customer(fake) for _ in range(SEED_CUSTOMERS)]
    cursor.executemany(
        """INSERT INTO customers
           (first_name, last_name, birth_date, street_address, city, state, zip_code, email, phone_number)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        customers,
    )

    logger.info("Seeding %d products", SEED_PRODUCTS)
    products = [make_product(fake) for _ in range(SEED_PRODUCTS)]
    cursor.executemany(
        "INSERT INTO products (sku, name, description, cost, in_stock_qty) VALUES (%s, %s, %s, %s, %s)",
        products,
    )

    # Backfill purchases across a rolling retention window ending today.
    today = date.today()
    start = today - timedelta(days=BACKFILL_DAYS)
    logger.info(
        "Backfilling %d purchases between %s and %s",
        BACKFILL_PURCHASES,
        start,
        today,
    )
    purchases = [
        make_purchase(
            SEED_CUSTOMERS,
            SEED_PRODUCTS,
            start + timedelta(days=random.randint(0, BACKFILL_DAYS)),
        )
        for _ in range(BACKFILL_PURCHASES)
    ]
    cursor.executemany(
        "INSERT INTO purchases (customer_id, product_id, qty_purchased, purchase_date) VALUES (%s, %s, %s, %s)",
        purchases,
    )

    conn.commit()
    logger.info("Initial seed and backfill complete")


# --------------------------------------------------------------------------------------
# Hourly insert + prune loop
# --------------------------------------------------------------------------------------


def tick(conn: pg8000.dbapi.Connection) -> None:
    """Insert a fresh batch of purchases dated today and prune old purchases."""
    max_customer, max_product = get_id_ranges(conn)
    if not max_customer or not max_product:
        logger.warning("No customers or products present; skipping insert")
        return

    cursor = conn.cursor()
    today = date.today()
    new_purchases = [
        make_purchase(max_customer, max_product, today) for _ in range(HOURLY_PURCHASES)
    ]
    cursor.executemany(
        "INSERT INTO purchases (customer_id, product_id, qty_purchased, purchase_date) VALUES (%s, %s, %s, %s)",
        new_purchases,
    )

    cursor.execute(
        "DELETE FROM purchases WHERE purchase_date < %s",
        (today - timedelta(days=RETENTION_DAYS),),
    )
    deleted = cursor.rowcount
    conn.commit()

    cursor.execute("SELECT count(*) FROM purchases")
    total = cursor.fetchone()[0]
    logger.info(
        "Inserted %d purchases, pruned %d older than %d days (total purchases: %d)",
        HOURLY_PURCHASES,
        deleted,
        RETENTION_DAYS,
        total,
    )


def main() -> None:
    conn = wait_for_database()
    try:
        initialize(conn)
        while True:
            try:
                tick(conn)
            except Exception:  # noqa: BLE001 - keep the loop alive, reconnect next tick
                logger.exception("Tick failed; reconnecting")
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
                conn = wait_for_database()
            time.sleep(INSERT_INTERVAL_SECONDS)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
