"""Batch-local deterministic text work; cancellation/results remain request-owned."""
from concurrent.futures import Future
from copy import deepcopy


def request_future(parent):
    child=Future()
    def complete(done):
        if not child.set_running_or_notify_cancel():return
        try:result=deepcopy(done.result())
        except BaseException as error:child.set_exception(error)
        else:child.set_result(result)
    parent.add_done_callback(complete)
    return child
