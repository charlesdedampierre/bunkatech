SHELL := /bin/zsh
.PHONY : all

help:
	cat Makefile

jupyter:
	python -m jupyterlab

streamlit:
	python -m streamlit run app.py