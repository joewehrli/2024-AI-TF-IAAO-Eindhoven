# 2024-AI-TF-IAAO-Eindhoven

## just enough container to get have miniconda and jupyter lab

```shell
cname=2024-ai-tf-oci
ename=ai-tf-lab
buildah --name $cname from public.ecr.aws/lts/ubuntu:24.04

buildah run $cname -- apt-get update

buildah run $cname -- apt-get install curl

buildah run $cname -- mkdir /root/tmp

buildah run $cname -- curl --output /root/tmp/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

buildah run $cname -- chmod +x /root/tmp/Miniconda3-latest-Linux-x86_64.sh

buildah run $cname -- ls -l /root/tmp

buildah run $cname -- /bin/bash /root/tmp/Miniconda3-latest-Linux-x86_64.sh -b

buildah run $cname -- /root/miniconda3/bin/conda init bash

buildah run $cname -- bash -i -c "conda --version"

buildah run $cname -- bash -i -c "conda  create --name $ename"

buildah run $cname -- bash -i -c "conda install -n $ename -c conda-forge jupyterlab"

buildah run $cname -- bash -i -c "conda env list"

buildah commit $cname $cname-devcon:0001
```

## how to run it

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root