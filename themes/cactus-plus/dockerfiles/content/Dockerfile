FROM ubuntu:14.04

RUN apt-get update -q && apt-get install --no-install-recommends -y ca-certificates git-core
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN git clone --recursive https://github.com/ceshine/ceshine-hugo.git /src/blog

COPY update.sh /src/

VOLUME ["/src/blog"]
WORKDIR /src/blog

# ENTRYPOINT ["git"]
CMD ["sh", "-c", "/src/update.sh"]
