# Safety backend image.
#
# Builds on the GDK dev image that's already cached on The Beast
# (roboservice/ros2-humble-gdk:latest) so we don't repeat the ROS2 +
# GDK install. We only add our code and pin the entrypoint.
#
# The image is self-contained: `docker run` with no volume mounts will
# launch the node from the code that was COPY'd in at build time. The
# accompanying docker-compose.yml additionally bind-mounts the source
# tree over the COPY'd files for fast development iteration.

FROM roboservice/ros2-humble-gdk:latest

ENV ROS_DOMAIN_ID=0 \
    ROS_LOCALHOST_ONLY=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# Python runtime dependencies (GDK pins protobuf; YOLO + vision stack on top).
COPY docker/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Bake the safety backend code into the image. The compose dev workflow
# bind-mounts the host copy over this — both paths are valid.
COPY safety_logic.py       /workspace/
COPY safety_loop.py        /workspace/
COPY safety_config.py      /workspace/
COPY safety_node.py        /workspace/
COPY safety_perception.py  /workspace/
COPY safety_backend_gdk.py /workspace/
COPY config.yaml           /workspace/
COPY offline.yaml          /workspace/

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Port 8080: MJPEG live stream (annotated color + depth)
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
# Default: ROS2 safety node (GDK perception + /safety/decision + /safety/stop + MJPEG :8080).
# Fallback (no ROS2): python3 safety_backend_gdk.py
# Offline/test:       python3 safety_backend_gdk.py --offline
CMD ["python3", "safety_node.py"]
