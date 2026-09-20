#!/bin/sh
# Re-create the 30-photo held-out background set (the images themselves are not redistributed here).
# Seeded URLs, so everyone gets the same photos.   Usage: sh fetch.sh [DEST_DIR]
D=${1:-full}; mkdir -p "$D"
for i in $(seq 1 30); do curl -sL --max-time 20 "https://picsum.photos/seed/yamheld$i/1600/900" -o "$D/novel_$i.jpg"; done
echo "fetched $(ls "$D" | wc -l) images into $D"
