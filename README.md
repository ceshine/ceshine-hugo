# Veritable Tech Blog

This repository contains the source code for the "Veritable Tech Blog," a blog by Ceshine Lee. The blog's tagline is "Technical Notes from a Data Geek."

## Running Locally

This blog is containerized using Docker. To run it locally, you need to have Docker and Docker Compose installed.

1.  Clone this repository.
2.  Run the following command in the root of the repository:

    ```bash
    docker-compose up
    ```

This will build the Docker images and start the Hugo server. You can then access the blog at `http://localhost:80`.

## Content

The content of the blog (the posts themselves) is not stored in this repository. It is located in a separate repository: [https://github.com/ceshine/ceshine-hugo.git](https://github.com/ceshine/ceshine-hugo.git).

The `content` service in the `docker-compose.yml` file clones this repository and keeps it up to date by running `git pull` every time the container starts.
