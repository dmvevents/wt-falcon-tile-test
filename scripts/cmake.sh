# # install cmake 
# mkdir /usr/src/server/opt
# apt remove cmake -y
# # cd /usr/src/server/opt
# # wget https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2.tar.gz
# # tar -zxvf cmake-3.19.2.tar.gz
# # cd cmake-3.19.2
# # ./bootstrap
# # make -j$(nproc)
# # make install
# # ldconfig
# apt purge --auto-remove cmake -y
# wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
# sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
# sudo apt update -y
# sudo apt install cmake -y
