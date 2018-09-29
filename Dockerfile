FROM jupyter/scipy-notebook
RUN pip install pandas-profiling

# copy all the files
RUN git clone "https://github.com/benmccary/federal_py"
COPY /data/ /home/jovyan/federal_py/data/

# make a new user
