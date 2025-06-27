# Development Instructions

This document describes how to start various development workflows.

## Developing for Google Colab

### Starting the Colab Local Runtime

Run the service in detached mode:

```bash
# Start `colab-local-env` container in the background
docker compose up -d colab-local-env
```

### Retrieving the Authentication URL

Once the container is up, extract the initial backend URL (with token) from the logs:

```bash
# Print the first URL matching the token pattern
docker compose logs colab-local-env | grep -m1 'http://127.0.0.1:9000/?token='
```

You can now copy and open the printed URL in your browser to authenticate.
