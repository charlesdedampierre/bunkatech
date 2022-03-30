FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip uninstall community
RUN pip install python-louvain
EXPOSE 8501
COPY . /app

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]