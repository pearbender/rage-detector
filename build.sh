#!/bin/bash
cd rage-gauge
docker build -t docker.fndk.io/other/rage-gauge-backend:latest .
docker push docker.fndk.io/other/rage-gauge-backend:latest

cd www
docker build -t docker.fndk.io/other/rage-gauge-front:latest .
docker push docker.fndk.io/other/rage-gauge-front:latest