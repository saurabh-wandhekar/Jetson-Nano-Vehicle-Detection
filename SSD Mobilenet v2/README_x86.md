# Instructions for x86_64 platforms

The demo in this repository, with minor tweaks, should also work on x86_64 platforms with NVIDIA GPU(s).  Here is a list of required modifications if you'd like to run the demos on an x86_64 PC/server.

Demo: SSD
-------------

1. Install `tensorrt`, `uff` and `graphsurgeon` python3 packages.  For example, I do the following on my x86_64 PC since I have TensorRT-5.1.5.0 installed at `/usr/local`.

   ```shell
   $ export TRT_DIR=/usr/local/TensorRT-5.1.5.0
   $ sudo pip3 install ${TRT_DIR}/python/tensorrt-5.1.5.0-cp36-none-linux_x86_64.whl
   $ sudo pip3 install ${TRT_DIR}/uff/uff-0.6.3-py2.py3-none-any.whl
   $ sudo pip3 install ${TRT_DIR}/graphsurgeon/graphsurgeon-0.4.1-py2.py3-none-any.whl
   ```

2. Patch `/usr/local/lib/python3.?/dist-packages/graphsurgeon/node_manipulation.py` by adding the following line (around line #42):

   ```python
    def shape(node):
        ......
        node.name = name or node.name
        node.op = op or node.op or node.name
   +    node.attr["dtype"].type = 1
        for key, val in kwargs.items():
        ......
   ```

3. Add TensorRT library path into `LD_LIBRARY_PATH` environment variable.  For example, I add the following line at the end of my `${HOME}/.bashrc` and log out/in again.

   ```
   export LD_LIBRARY_PATH=/usr/local/TensorRT-5.1.5.0/lib\
                          ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```

4. Re-build `libflattenconcat.so` from TensorRT's 'python/uff_ssd' sample source code.  For example,

   ```shell
   $ mkdir -p ${HOME}/src/TensorRT-5.1.5.0
   $ cp -r /usr/local/TensorRT-5.1.5.0/samples ${HOME}/src/TensorRT-5.1.5.0
   $ cd ${HOME}/src/TensorRT-5.1.5.0/samples/python/uff_ssd
   $ mkdir build
   $ cd build
   $ cmake -D NVINFER_LIB=/usr/local/TensorRT-5.1.5.0/lib/libnvinfer.so \
           -D TRT_INCLUDE=/usr/local/TensorRT-5.1.5.0/include ..
   $ make
   $ cp libflattenconcat.so ${HOME}/project/tensorrt_demos/ssd/
   ```

5. Install PyCUDA.

   ```shell
   $ sudo apt-get install -y build-essential python-dev
   $ sudo apt-get install -y libboost-python-dev libboost-thread-dev
   $ sudo pip3 install setuptools
   $ export boost_pylib=$(basename /usr/lib/x86_64-linux-gnu/libboost_python3-py3?.so)
   $ export boost_pylibname=${boost_pylib%.so}
   $ export boost_pyname=${boost_pylibname/lib/}
   $ cd ${HOME}/src
   $ wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
   $ tar xzvf pycuda-2019.1.2.tar.gz
   $ cd pycuda-2019.1.2
   $ ./configure.py --python-exe=/usr/bin/python3 \
                    --cuda-root=/usr/local/cuda \
                    --cudadrv-lib-dir=/usr/lib/x86_64-linux-gnu \
                    --boost-inc-dir=/usr/include \
                    --boost-lib-dir=/usr/lib/x86_64-linux-gnu \
                    --boost-python-libname=${boost_pyname} \
                    --boost-thread-libname=boost_thread \
                    --no-use-shipped-boost
   $ make -j4
   $ python3 setup.py build
   $ sudo python3 setup.py install
   $ python3 -c "import pycuda; print('pycuda version:', pycuda.VERSION)"
   ```

6. Follow the steps in the original [README.md](https://github.com/jkjung-avt/tensorrt_demos/blob/master/README.md) but skip `install.sh`.  You should be able to build the SSD TensorRT engines and run them on on x86_64 as well.
