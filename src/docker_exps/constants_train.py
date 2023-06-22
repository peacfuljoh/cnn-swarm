
from queue import Queue


TRAIN_STATUS = dict(
    jobs={} # Dict[dict], key: job_id, sub-keys: duration, progress (%), complete (bool)
)

JOBS_INIT_INFO = dict(id_latest=0)

JOB_MSG_QUEUE = Queue()

TRAIN_TASKS = {}