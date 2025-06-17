# db_utils.py

import os
import datetime
import ssl
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = "chimney_db"
EVENTS    = "yellow_gas_events"

# ─── Persistent client ───────────────────────────────────────────────
try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tls=True,
        tlsAllowInvalidCertificates=True
    )
    client.admin.command("ping")
except (ConnectionFailure, ConfigurationError) as e:
    raise RuntimeError(f"MongoDB connection failed: {e}")

db = client[DB_NAME]
events_coll = db[EVENTS]

def get_db_collection():
    return events_coll

def insert_event_start(chimney_number, start_time):
    doc = {
        "chimney_number": int(chimney_number),
        "start_time":      float(start_time),
        "end_time":        None,
        "added_on":        datetime.datetime.utcnow()
    }
    return events_coll.insert_one(doc).inserted_id

def update_event_end(event_id, end_time):
    return events_coll.update_one(
        {"_id": event_id},
        {"$set": {"end_time": float(end_time)}}
    ).modified_count
