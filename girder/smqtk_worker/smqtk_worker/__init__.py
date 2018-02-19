from girder_worker import GirderWorkerPluginABC


class GirderWorkerPlugin(GirderWorkerPluginABC):
    def __init__(self, app, *args, **kwargs):
        self.app = app

    def task_imports(self):
        return ['smqtk_worker.tasks']
