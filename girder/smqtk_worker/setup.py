import setuptools

setuptools.setup(
    name='smqtk_worker',
    version='0.0.1',
    description='A Girder Worker plugin for distributed processing of images using SMQTK.',
    author='Kitware, Inc.',
    author_email='kitware@kitware.com',
    license='Apache 2.0',
    entry_points={
        'girder_worker_plugins': [
            'smqtk_worker = smqtk_worker:GirderWorkerPlugin'
        ]
    }
)
