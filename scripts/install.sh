mkdir /usr/src/server/opt
# install libmodbus
cd /usr/src/server/opt
git clone https://github.com/stephane/libmodbus.git
cd libmodbus
git submodule update --init --recursive
./autogen.sh
./configure && make install
sudo ldconfig

# install curl for the people
cd /usr/src/server/opt
git clone https://github.com/whoshuu/cpr.git
cd cpr 
git submodule update --init --recursive
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j
sudo make install

sudo ln -sf /usr/src/server/opt/cpr/build/lib/libcurl-d.so /usr/local/lib/
sudo ldconfig

