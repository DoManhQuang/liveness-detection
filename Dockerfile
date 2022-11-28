#FROM python:3.8
#
## install the dependencies and packages in the requirements file
#RUN pip install -r requirements.txt
#
#RUN mkdir app
#
## switch working directory
#WORKDIR /app
#
## copy every content from the local file to the image
#COPY . /app
#
## configure the container to run in an executed manner
#ENTRYPOINT [ "python3" ]
#
#CMD ["../tools/.py"]
