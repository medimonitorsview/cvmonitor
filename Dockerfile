FROM ubuntu:18.04
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install -yy wget libzbar0 libjpeg-turbo8-dev libz-dev python3-pip python3-venv git python3-tk &&  rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -mvenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /opt/app/
RUN pip install -U wheel setuptools_scm setuptools
ADD requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt
COPY . /opt/app
COPY .git /opt/app/.git
RUN pip install -e . --no-cache-dir
RUN pytest
CMD cvmonitor
