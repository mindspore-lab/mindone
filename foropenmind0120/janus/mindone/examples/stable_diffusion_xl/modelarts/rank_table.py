import json
import os
import time

from common import ModelArts, RankTableEnv, RunAscendLog

log = RunAscendLog.get_run_ascend_logger()


class Device:
    def __init__(self, device_id, device_ip, rank_id):
        self.device_id = device_id
        self.device_ip = device_ip
        self.rank_id = rank_id


class Instance:
    def __init__(self, pod_name, server_id, devices):
        self.pod_name = pod_name
        self.server_id = server_id
        self.devices = self.parse_devices(devices)

    @staticmethod
    def parse_devices(devices):
        if devices is None:
            return []
        device_object_list = []
        for device in devices:
            device_object_list.append(Device(device["device_id"], device["device_ip"], ""))

        return device_object_list

    def set_devices(self, devices):
        self.devices = devices


class Group:
    def __init__(self, group_name, device_count, instance_count, instance_list):
        self.group_name = group_name
        self.device_count = int(device_count)
        self.instance_count = int(instance_count)
        self.instance_list = self.parse_instance_list(instance_list)

    @staticmethod
    def parse_instance_list(instance_list):
        instance_object_list = []
        for instance in instance_list:
            instance_object_list.append(Instance(instance["pod_name"], instance["server_id"], instance["devices"]))

        return instance_object_list


class RankTable:
    STATUS_FIELD = "status"
    COMPLETED_STATUS = "completed"

    def __init__(self):
        self.rank_table_path = ""
        self.rank_table = {}

    @staticmethod
    def read_from_file(file_path):
        with open(file_path) as json_file:
            return json.load(json_file)

    @staticmethod
    def wait_for_available(rank_table_file, period=1):
        log.info("Wait for Rank table file at %s ready" % rank_table_file)
        complete_flag = False
        while not complete_flag:
            with open(rank_table_file) as json_file:
                data = json.load(json_file)
                if data[RankTable.STATUS_FIELD] == RankTable.COMPLETED_STATUS:
                    log.info("Rank table file is ready for read")
                    log.info("\n" + json.dumps(data, indent=4))
                    return True

            time.sleep(period)

        return False

    @staticmethod
    def convert_server_to_instance(server):
        device_list = []
        for device in server["device"]:
            device_list.append(
                Device(device_id=device["device_id"], device_ip=device["device_ip"], rank_id=device["rank_id"])
            )

        ins = Instance(pod_name="", server_id=server["server_id"], devices=[])
        ins.set_devices(device_list)
        return ins

    def get_rank_table_path(self):
        return self.rank_table_path

    def get_server(self, server_id):
        for server in self.rank_table["server_list"]:
            if server["server_id"] == server_id:
                log.info("Current server")
                log.info("\n" + json.dumps(server, indent=4))
                return server

        log.error("server [%s] is not found" % server_id)
        return None


class RankTableTemplate2(RankTable):
    def __init__(self, rank_table_template2_path):
        super().__init__()

        json_data = self.read_from_file(file_path=rank_table_template2_path)

        self.status = json_data[RankTableTemplate2.STATUS_FIELD]
        if self.status != RankTableTemplate2.COMPLETED_STATUS:
            return

        # sorted instance list by the index of instance
        # assert there is only one group
        json_data["group_list"][0]["instance_list"] = sorted(
            json_data["group_list"][0]["instance_list"], key=RankTableTemplate2.get_index
        )

        self.group_count = int(json_data["group_count"])
        self.group_list = self.parse_group_list(json_data["group_list"])

        self.rank_table_path, self.rank_table = self.convert_template2_to_template1_format_file()

    @staticmethod
    def parse_group_list(group_list):
        group_object_list = []
        for group in group_list:
            group_object_list.append(
                Group(group["group_name"], group["device_count"], group["instance_count"], group["instance_list"])
            )

        return group_object_list

    @staticmethod
    def get_index(instance):
        # pod_name example: job94dc1dbf-job-bj4-yolov4-15
        pod_name = instance["pod_name"]
        return int(pod_name[pod_name.rfind("-") + 1 :])

    def get_current_instance(self):
        """
        get instance by pod name
        specially, return the first instance when the pod name is None
        :return:
        """
        pod_name = ModelArts.get_current_instance_name()
        if pod_name is None:
            if len(self.group_list) > 0:
                if len(self.group_list[0].instance_list) > 0:
                    return self.group_list[0].instance_list[0]

            return None

        for group in self.group_list:
            for instance in group.instance_list:
                if instance.pod_name == pod_name:
                    return instance
        return None

    def convert_template2_to_template1_format_file(self):
        rank_table_template1_file = {"status": "completed", "version": "1.0", "server_count": "0", "server_list": []}

        logic_index = 0
        server_map = {}
        # collect all devices in all groups
        for group in self.group_list:
            if group.device_count == 0:
                continue
            for instance in group.instance_list:
                if instance.server_id not in server_map:
                    server_map[instance.server_id] = []

                for device in instance.devices:
                    template1_device = {
                        "device_id": device.device_id,
                        "device_ip": device.device_ip,
                        "rank_id": str(logic_index),
                    }
                    logic_index += 1
                    server_map[instance.server_id].append(template1_device)

        server_count = 0
        for server_id in server_map:
            rank_table_template1_file["server_list"].append({"server_id": server_id, "device": server_map[server_id]})
            server_count += 1

        rank_table_template1_file["server_count"] = str(server_count)

        log.info("Rank table file (Template1)")
        log.info("\n" + json.dumps(rank_table_template1_file, indent=4))

        if not os.path.exists(RankTableEnv.get_rank_table_template1_file_dir()):
            os.makedirs(RankTableEnv.get_rank_table_template1_file_dir())

        path = os.path.join(RankTableEnv.get_rank_table_template1_file_dir(), RankTableEnv.HCCL_JSON_FILE_NAME)
        with open(path, "w") as f:
            f.write(json.dumps(rank_table_template1_file))
            log.info("Rank table file (Template1) is generated at %s", path)

        return path, rank_table_template1_file

    def get_device_num(self):
        total_device_num = 0
        for group in self.group_list:
            total_device_num += group.device_count
        return total_device_num


class RankTableTemplate1(RankTable):
    def __init__(self, rank_table_template1_path):
        super().__init__()
        self.rank_table_path = rank_table_template1_path
        self.rank_table = self.read_from_file(file_path=rank_table_template1_path)

    def get_current_instance(self):
        current_server = None
        server_list = self.rank_table["server_list"]
        if len(server_list) == 1:
            current_server = server_list[0]
        elif len(server_list) > 1:
            host_ip = ModelArts.get_current_host_ip()
            if host_ip is not None:
                for server in server_list:
                    if server["server_id"] == host_ip:
                        current_server = server
                        break
            else:
                current_server = server_list[0]

        if current_server is None:
            log.error("server is not found")
            return None
        return self.convert_server_to_instance(current_server)

    def get_device_num(self):
        server_list = self.rank_table["server_list"]
        device_num = 0
        for server in server_list:
            device_num += len(server["device"])
        return device_num
