# ARG UBUNTU_VERSION=18.04
# FROM ubuntu:$UBUNTU_VERSION

# ARG PYTHON_VERSION=2.7.5

# # Install dependencies
# # PIP - openssl version > 1.1 may be an issue (try older ubuntu images)
# RUN apt-get update \
#   && apt-get install -y wget gcc make openssl libffi-dev libgdbm-dev libsqlite3-dev libssl-dev zlib1g-dev \
#   && apt-get clean

# WORKDIR /tmp/

# # Build Python from source
# RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz \
#   && tar --extract -f Python-$PYTHON_VERSION.tgz \
#   && cd ./Python-$PYTHON_VERSION/ \
#   && ./configure --enable-optimizations --prefix=/usr/local \
#   && make && make install \
#   && cd ../ \
#   && rm -r ./Python-$PYTHON_VERSION*

# RUN python --version

FROM rtikid/python2-numpy-scipy-sympy-neuron-brian2-netpyne-inspyred-pyabf:latest
WORKDIR /code
COPY /src /code
# copy specific file due to ignore file prefix with [dot]
COPY /src/.env /code/.env

#COPY ./requirements.txt /code/requirements.txt
#COPY ./test.py /code/test.py

# RUN apt-get update -y
# RUN apt-get install -y libhdf5-dev
RUN pip install wheel
RUN pip install --user -r requirements.txt
#RUN --mount=type=bind,source=fonts.hdf5,target=/code/fonts.hdf5
EXPOSE 5000
CMD [ "python", "-c", "'print u\"Hello World\"'" ]  
CMD [ "python", "real_vs_pred.py" ]  
CMD [ "python", "server.py" ]  
#CMD [ "python", "test.py" ]  

