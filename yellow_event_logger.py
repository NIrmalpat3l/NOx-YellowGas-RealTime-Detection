# yellow_event_logger.py

import time
from db_utils import insert_event_start, update_event_end

class YellowGasEventLogger:
    def __init__(self):
        self.active_events = {}

    def update(self, yellow_flags: dict, timestamp: float = None):
        ts = timestamp if timestamp is not None else time.time()

        # start new
        for cid, flag in yellow_flags.items():
            if flag and cid not in self.active_events:
                eid = insert_event_start(cid, ts)
                self.active_events[cid] = eid
                print(f"[LOGGER] START chimney {cid} @ {ts}")

        # end old
        for cid in list(self.active_events):
            if not yellow_flags.get(cid, False):
                eid = self.active_events.pop(cid)
                update_event_end(eid, ts)
                print(f"[LOGGER] END   chimney {cid} @ {ts}")

    def close_all(self, timestamp: float = None):
        ts = timestamp if timestamp is not None else time.time()
        for cid, eid in self.active_events.items():
            update_event_end(eid, ts)
            print(f"[LOGGER] FORCE-END chimney {cid} @ {ts}")
        self.active_events.clear()
