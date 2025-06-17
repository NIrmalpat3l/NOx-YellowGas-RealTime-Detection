# yellow_gas_summary_collector.py

import os
from pymongo import UpdateOne
from db_utils import get_db_collection  # :contentReference[oaicite:2]{index=2}

def collect_summary():
    """
    Reads all completed yellow-gas events (where end_time != None) and
    aggregates total duration per chimney per day, then upserts into
    'yellow_gas_summary' collection.
    """
    raw_coll = get_db_collection()
    db = raw_coll.database
    summary_coll = db['yellow_gas_summary']

    # Pipeline:
    # 1) only closed events (end_time != null)
    # 2) compute duration = end_time - start_time
    # 3) convert start_time (seconds since epoch) to a Date, then format as YYYY-MM-DD
    # 4) group by chimney_number + day, summing durations
    pipeline = [
        {"$match": {"end_time": {"$ne": None}}},
        {"$project": {
            "chimney_number": 1,
            "duration": {"$subtract": ["$end_time", "$start_time"]},
            # convert start_time in seconds to millis, then to Date
            "day": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": {"$toDate": {"$multiply": ["$start_time", 1000]}}
                }
            }
        }},
        {"$group": {
            "_id": {"chimney_number": "$chimney_number", "day": "$day"},
            "total_duration": {"$sum": "$duration"}
        }},
        {"$project": {
            "_id": 0,
            "chimney_number": "$_id.chimney_number",
            "day": "$_id.day",
            "total_duration": 1
        }}
    ]

    cursor = raw_coll.aggregate(pipeline)

    # Prepare bulk upserts into summary_coll
    requests = []
    for doc in cursor:
        chimney = doc['chimney_number']
        day     = doc['day']
        total   = doc['total_duration']
        # upsert by chimney + day
        requests.append(
            UpdateOne(
                {"chimney_number": chimney, "day": day},
                {"$set": {"total_duration": total}},
                upsert=True
            )
        )

    if requests:
        result = summary_coll.bulk_write(requests)
        print(f"Upserted {result.upserted_count} docs, modified {result.modified_count} docs.")
    else:
        print("No completed events to summarize.")

if __name__ == "__main__":
    """
    Running this script will rebuild the 'yellow_gas_summary' collection
    based on current raw events.
    """
    collect_summary()
