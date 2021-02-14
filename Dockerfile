FROM ubuntu:20.04
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev
WORKDIR ./app
EXPOSE 8080
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["run.py"]