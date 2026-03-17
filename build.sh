cd sampling_server && \
make clean && make -j 8 && \
cd .. && \
cd training_backend && \
CUDA_VISIBLE_DEVICES=0 python setup.py install && \
cd ..