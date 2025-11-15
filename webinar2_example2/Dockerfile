FROM jupyter/datascience-notebook:notebook-6.5.4

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD start-notebook.sh --NotebookApp.token='' --port=9999