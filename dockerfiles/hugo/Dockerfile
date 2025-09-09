FROM ubuntu:24.04

RUN apt-get update -q && apt-get install -y curl
RUN curl -L -o /tmp/hugo.tar.gz https://github.com/gohugoio/hugo/releases/download/v0.149.1/hugo_0.149.1_Linux-64bit.tar.gz
RUN tar zxvf /tmp/hugo.tar.gz -C /tmp  && mv /tmp/hugo /usr/local/bin/hugo

# VOLUME ["/var/www/blog"]

ENTRYPOINT ["hugo"]
#CMD ["-w", "-s", "/src/blog", "-d", "/var/www/blog"]
CMD ["server", "-w", "-s", "/src/blog", "--bind=0.0.0.0", "--appendPort=false", "-b", "https://blog.ceshine.net", "--disableLiveReload"]
