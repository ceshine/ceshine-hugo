FROM alpine:3.18

ARG HUGO_VERSION=0.149.1
# The extended version has built-in SASS/SCSS support.
ARG HUGO_BINARY=hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz

# Install Hugo
RUN apk add --no-cache curl \
    && curl -L -o /tmp/hugo.tar.gz https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/${HUGO_BINARY} \
    && tar zxvf /tmp/hugo.tar.gz -C /usr/local/bin \
    && rm /tmp/hugo.tar.gz \
    && apk del curl

# Set the working directory
WORKDIR /src

# Expose the default Hugo port
EXPOSE 1313

# Start the Hugo server, building drafts and unpublished content
CMD ["server", "--bind=0.0.0.0", "-D"]
