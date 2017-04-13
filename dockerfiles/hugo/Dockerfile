FROM ubuntu:14.04

RUN apt-get update -q && apt-get install -y curl
RUN curl -L -o /tmp/hugo.tar.gz https://github.com/spf13/hugo/releases/download/v0.20/hugo_0.20_Linux-64bit.tar.gz
RUN tar zxvf /tmp/hugo.tar.gz -C /tmp  && mv /tmp/hugo_0.20_linux_amd64/hugo_0.20_linux_amd64 /usr/local/bin/hugo

# VOLUME ["/var/www/blog"]

RUN apt-get install -y python-pygments
ENTRYPOINT ["hugo"]
#CMD ["-w", "-s", "/src/blog", "-d", "/var/www/blog"]
CMD ["server", "-w", "-s", "/src/blog", "--bind=0.0.0.0", "--appendPort=false", "-v", "-b", "https://blog.ceshine.net", "--disableLiveReload"]
