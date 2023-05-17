from fabric import task, Connection

@task
def install(c):
    if not c.run('test -f latent-diffusion', warn=True).failed:
        c.run('git clone https://github.com/adammoss/latent-diffusion')
    with c.cd('latent-diffusion'):
        c.run('conda env create -f environment.yaml')
        c.run('conda activate ldm')
        c.run('pip install protobuf==3.20.*')
        c.run('sudo apt-get install git-lfs')
        c.run('git lfs install')

@task
def install_conda(c):
    c.run('wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh')
    c.run('bash Anaconda3-2023.03-1-Linux-x86_64.sh')
