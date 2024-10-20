source ./build-buildah-oci.conf

cdeps=""
pdeps="\
jupyterlab \
plotly \
scikit-learn \
lightgbm \
pyproj \
geopandas \
matplotlib \
seaborn \
openpyxl \
"

#tensorflow-cpu=2.10.0 \
echo cname=$cname
echo ename=$ename
echo pver=$pver
echo cdeps=$cdeps
echo pdeps=$pdeps

make_pip_container(){
buildah --name $cname from public.ecr.aws/lts/ubuntu:24.04
buildah run $cname -- bash -c "DEBIAN_FRONTEND=noninteractive apt-get -y update"
}

make_pip_env() {
buildah run $cname -- bash -i -c "apt-get install -y pip --update"
buildah run $cname -- bash -i -c "pip install $pdeps --break-system-packages"
}

#make_conda_container
#make_conda_env

#buildah commit $cname $cname-devcon:0001
#buildah commit $cname $cname-devcon:0002
#buildah commit $cname $cname-devcon:0003
#buildah commit $cname $cname-devcon:0004
## works!
#buildah commit $cname $cname-devcon:0005

# jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# pip install tensorflow --break-system-packages
