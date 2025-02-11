import os
import os.path
import signal
import time

from common import RunAscendLog
from fmk import FMK

log = RunAscendLog.get_run_ascend_logger()


class FMKManager:
    # max destroy time: ~20 (15 + 5)
    # ~ 15 (1 + 2 + 4 + 8)
    MAX_TEST_PROC_CNT = 4

    def __init__(self, instance):
        self.instance = instance
        self.fmk = []
        self.fmk_processes = []
        self.get_sigterm = False
        self.max_test_proc_cnt = FMKManager.MAX_TEST_PROC_CNT

    # break the monitor and destroy processes when get terminate signal
    def term_handle(func):
        def receive_term(signum, stack):
            log.info("Received terminate signal %d, try to destroyed all processes" % signum)
            stack.f_locals["self"].get_sigterm = True

        def handle_func(self, *args, **kwargs):
            origin_handle = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, receive_term)
            res = func(self, *args, **kwargs)
            signal.signal(signal.SIGTERM, origin_handle)
            return res

        return handle_func

    def run(self, rank_size, command):
        for index, device in enumerate(self.instance.devices):
            fmk_instance = FMK(index, device)
            self.fmk.append(fmk_instance)

            self.fmk_processes.append(fmk_instance.run(rank_size, command))

    @term_handle
    def monitor(self, period=1):
        # busy waiting for all fmk processes exit by zero
        # or there is one process exit by non-zero

        fmk_cnt = len(self.fmk_processes)
        zero_ret_cnt = 0
        while zero_ret_cnt != fmk_cnt:
            zero_ret_cnt = 0
            for index in range(fmk_cnt):
                fmk = self.fmk[index]
                fmk_process = self.fmk_processes[index]
                if fmk_process.poll() is not None:
                    if fmk_process.returncode != 0:
                        log.error(
                            "proc-rank-%s-device-%s (pid: %d) has exited with non-zero code: %d"
                            % (fmk.rank_id, fmk.device_id, fmk_process.pid, fmk_process.returncode)
                        )
                        return fmk_process.returncode

                    zero_ret_cnt += 1
            if self.get_sigterm:
                break
            time.sleep(period)

        return 0

    def destroy(self, base_period=1):
        log.info("Begin destroy training processes")
        self.send_sigterm_to_fmk_process()
        self.wait_fmk_process_end(base_period)
        log.info("End destroy training processes")

    def send_sigterm_to_fmk_process(self):
        # send SIGTERM to fmk processes (and process group)
        for r_index in range(len(self.fmk_processes) - 1, -1, -1):
            fmk = self.fmk[r_index]
            fmk_process = self.fmk_processes[r_index]
            if fmk_process.poll() is not None:
                log.info(
                    "proc-rank-%s-device-%s (pid: %d) has exited before receiving the term signal",
                    fmk.rank_id,
                    fmk.device_id,
                    fmk_process.pid,
                )
                del self.fmk_processes[r_index]
                del self.fmk[r_index]

            try:
                os.killpg(fmk_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def wait_fmk_process_end(self, base_period):
        test_cnt = 0
        period = base_period
        while len(self.fmk_processes) > 0 and test_cnt < self.max_test_proc_cnt:
            for r_index in range(len(self.fmk_processes) - 1, -1, -1):
                fmk = self.fmk[r_index]
                fmk_process = self.fmk_processes[r_index]
                if fmk_process.poll() is not None:
                    log.info("proc-rank-%s-device-%s (pid: %d) has exited", fmk.rank_id, fmk.device_id, fmk_process.pid)
                    del self.fmk_processes[r_index]
                    del self.fmk[r_index]
            if not self.fmk_processes:
                break

            time.sleep(period)
            period *= 2
            test_cnt += 1

        if len(self.fmk_processes) > 0:
            for r_index in range(len(self.fmk_processes) - 1, -1, -1):
                fmk = self.fmk[r_index]
                fmk_process = self.fmk_processes[r_index]
                if fmk_process.poll() is None:
                    log.warn(
                        "proc-rank-%s-device-%s (pid: %d) has not exited within the max waiting time, "
                        "send kill signal",
                        fmk.rank_id,
                        fmk.device_id,
                        fmk_process.pid,
                    )
                    os.killpg(fmk_process.pid, signal.SIGKILL)
