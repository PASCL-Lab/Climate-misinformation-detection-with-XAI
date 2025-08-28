# TODO: Modify this Procfile to fit your needs
web: gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:$PORT --timeout 300 --preload --log-level info main:app
