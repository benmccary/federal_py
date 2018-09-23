FROM jupyter/scipy-notebook
RUN pip install pandas-profiling

# copy all the files
COPY . /home/jovyan/work/federal

# make a new user
