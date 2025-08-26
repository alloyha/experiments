FROM public.ecr.aws/lambda/python:3.9.2023.03.15.15-x86_64

# Set the working directory to the parent directory of Python scripts and pyproject.toml
WORKDIR ${LAMBDA_TASK_ROOT}

# Install system dependencies (if needed)
RUN pip install poetry

# Install system dependencies (if needed)
RUN pip install poetry

# Copy your application code
COPY . .

# Set the CMD to your handler
CMD ["lambda_api.predict_service.predict"]