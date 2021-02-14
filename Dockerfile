FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
WORKDIR /app
COPY . /app
COPY requirements.txt .
EXPOSE 80
RUN apt-get update
#RUN apt-get -y install python3-dev
RUN pip --no-cache-dir install -r requirements.txt
CMD uvicorn main:app --host 0.0.0.0 --port 80 --reload