FROM python:3.12-slim
WORKDIR /opt/sls-scan
RUN pip install --no-cache-dir numpy==2.5.4 opencv-contrib-python-headless==5.0.0.93 pyyaml==6.0.3
COPY pyproject.toml README.md ./
COPY slscan ./slscan
RUN pip install --no-cache-dir --no-deps .
EXPOSE 5600
ENTRYPOINT ["slscan", "node-sim", "--host", "0.0.0.0", "--port", "5600"]
CMD ["--scene", "plate"]
