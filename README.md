# python_docker_case

Part 1: Docker
1. Create an empty repository and initialize it with git. Please pay attention to a proper
commit log along the case.
2. Create a “dummy” main.py file, which will be the starting point for the Python app. This
file will be filled in Part 2 and 3.
3. Create a Dockerfile that can be used to build a Docker image for Python 3.7, that
copies all required project files into the image, including a requirements.txt file that is
“pip install”-ed during the build of the image.
4. When running the Docker image, the main.py file is supposed to be executed as the
CMD.