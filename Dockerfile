FROM python:3.13

ENV PYTHONUNBUFFERED=1 \
    TQDM_POSITION=-1

WORKDIR /app

# Versions pinned to match uv.lock, so the image doesn't silently drift
# from whatever's actually been tested. Keep these in sync if uv.lock changes.
#
# CPU-only torch: the default PyPI wheel drags in ~5GB of unused CUDA
# dependencies. This script only does small equirect->perspective tensor
# ops on CPU, so the CPU wheel is all that's needed here.
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        numpy==2.4.2 \
        pandas==3.0.0 \
        pillow==12.1.0 \
        pytorch360convert==0.2.3 \
        streetlevel==0.12.11 \
        tqdm==4.67.3

COPY salty_image_grabber.py .

ENTRYPOINT ["python", "salty_image_grabber.py"]
