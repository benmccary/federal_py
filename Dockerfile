FROM jupyter/scipy-notebook
RUN pip install pandas-profiling
# it might be good to add something that will create a user for me
# instead of using jovyan
