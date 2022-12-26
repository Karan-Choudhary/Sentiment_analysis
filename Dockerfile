FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

# FROM public.ecr.aws/lambda/python:3.8

# # Install the function's dependencies using file requirements.txt
# # from your project folder.

# COPY app ./app

# COPY requirements.txt  .
# RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# # Copy function code
# COPY app/main.py ${LAMBDA_TASK_ROOT}

# # Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
# CMD [ "app.handler" ]