source ./build-buildah-oci.conf

cdeps="\
jupyterlab \
scikit-learn \
lightgbm \
pyproj \
geopandas \
matplotlib \
seaborn \
openpyxl \
"

#tensorflow-cpu=2.10.0 \

pdeps="plotly.express"
echo cname=$cname
echo ename=$ename
echo pver=$pver
echo pdeps=$cdeps
echo pdeps=$pdeps

make_conda_container(){

buildah --name $cname from public.ecr.aws/lts/ubuntu:24.04

buildah run $cname -- bash -c "DEBIAN_FRONTEND=noninteractive apt-get -y update"

buildah run $cname -- apt-get install -y curl

buildah run $cname -- mkdir /root/tmp

buildah run $cname -- curl --output /root/tmp/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

buildah run $cname -- chmod +x /root/tmp/Miniconda3-latest-Linux-x86_64.sh

buildah run $cname -- ls -l /root/tmp

buildah run $cname -- /bin/bash /root/tmp/Miniconda3-latest-Linux-x86_64.sh -b

buildah run $cname -- /root/miniconda3/bin/conda init bash

buildah run $cname -- bash -i -c "conda --version"

buildah run $cname -- bash -i -c "conda env list"
}

make_conda_env() {

buildah run $cname -- bash -i -c "conda  create -y --name $ename python=$pyver"

buildah run $cname -- bash -i -c "conda --version"

buildah run $cname -- bash -i -c "conda  create -y --name $ename python=$pyver"

buildah run $cname -- bash -i -c "conda env list"

buildah run $cname -- bash -i -c "conda install -y -n $ename -c conda-forge $cdeps"

#buildah run $cname -- bash -i -c "conda run -n $ename pip install $pdeps"
}

#make_conda_container
make_conda_env

#buildah commit $cname $cname-devcon:0001

# buildah run $cname -- bash -i -c "conda env remove -n $ename"

# buildah run $cname -- bash -i -c "conda run -n $ename jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"