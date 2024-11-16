FROM python:3.7-stretch

ADD requirements.txt /.
RUN pip install -r /requirements.txt

ADD . /code/

WORKDIR /code

CMD ["/code/get_service_opioid.py"]
ENTRYPOINT ["python"]

#ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]