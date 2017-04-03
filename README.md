# TranSparse and TensorFlow
Implement the model of TranSparse described in  [Ji et al., 2016](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11982/11693)

Training:
```bash
# compile c++ op:
$ TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
$ g++ -std=c++11 -shared norm_prjct_op.cc norm_prjct_kernel.cc -o \
       norm_prjct_op.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
$ python3 transparse.py # Train
$ python evaluate.py    # Evaluate
```

Test and debug:
```bash
$ python3 transparse.py --test --debug
```
