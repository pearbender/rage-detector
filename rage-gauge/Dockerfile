FROM python:3.12

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y ffmpeg

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["python3", "-u", "main.py"]
