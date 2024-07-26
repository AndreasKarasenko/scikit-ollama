# import json
# import subprocess


# class OllamaServiceManager:
#     """A support class that manages the underlying Ollama server.

#     Provides information and can start / stop / restart the server
#     """

#     def __init__(self, host: str = "http://localhost:11434"):
#         self.service_name = "ollama"
#         self.host = host

#     def _run_command(self, command):
#         """Runs a subprocess command.

#         Parameters
#         ----------
#         command : str
#             command to run

#         Returns
#         -------
#         None
#         """
#         try:
#             result = subprocess.run(
#                 command,
#                 shell=True,
#                 check=True,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#             )
#             return result.stdout.decode("utf-8")
#         except subprocess.CalledProcessError as e:
#             return e.stderr.decode("utf-8")

#     def start_service(self):
#         return self._run_command(f"sudo systemctl start {self.service_name}")

#     def stop_service(self):
#         return self._run_command(f"sudo systemctl stop {self.service_name}")

#     def service_status(self):
#         return self._run_command(f"sudo systemctl status {self.service_name}")

#     def pull_model(self, model: str):
#         return self._run_command(f"{self.service_name} pull {model}")

#     def get_models(self):
#         models = json.loads(self._run_command("curl http://localhost:11434/api/tags"))
#         models = [i["name"] for i in models["models"]]
#         return models

#     def service_status(self):
#         try:
#             result = subprocess.run(
#                 "ollama ps",
#                 shell=True,
#                 check=True,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#             )
#             return True
#         except subprocess.CalledProcessError:
#             return False
