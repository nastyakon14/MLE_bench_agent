FROM mlebench-env

RUN conda create -n green python=3.11 -y
COPY requirements.txt /tmp/green-requirements.txt
RUN conda run -n green pip install -r /tmp/green-requirements.txt && \
    conda run -n green pip install -e /mlebench

RUN mkdir cache && \
    chown nonroot cache

COPY src /home/green/src

USER nonroot

RUN --mount=type=secret,id=kaggle_json,target=/home/nonroot/.config/kaggle/kaggle.json,uid=1000,gid=1000 \
    conda run -n green mlebench prepare -c spaceship-titanic

ENTRYPOINT ["/opt/conda/envs/green/bin/python", "/home/green/src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009