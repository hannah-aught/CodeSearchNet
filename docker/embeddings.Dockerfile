FROM python:3.6
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 

RUN pip --no-cache-dir install --upgrade \
    gensim==3.2.0 \
    numpy==1.16.5 \ 
    scipy==1.3.1

WORKDIR /home/dev/src

CMD bash