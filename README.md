# Veritable Tech Blog

This repository contains the source code for the "Veritable Tech Blog," a blog by Ceshine Lee. The blog's tagline is "Technical Notes from a Data Geek."

The content of the blog is located in the `content` directory of this repository.

## Running Locally

To run the blog locally, you need to have [Hugo](https://gohugo.io/getting-started/installing/) installed.

1.  Clone this repository.
2.  Run the following command in the root of the repository:

    ```bash
    hugo server
    ```

This will start the Hugo development server, and you can view the blog at `http://localhost:1313`.

## Deployment

The Docker setup (`docker-compose.yml` and `dockerfiles/`) in this repository is intended for deployment on a VPS (Virtual Private Server) and is not needed for local development.
