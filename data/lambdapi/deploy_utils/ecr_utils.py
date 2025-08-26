from os import system
import subprocess
from subprocess import DEVNULL

def build_tagged_image(image_name, tag):
    return f"{image_name}:{tag}"

def build_ecr_password_stdin(account_id_, region_):
    return f"{account_id_}.dkr.ecr.{region_}.amazonaws.com"

def build_ecr_url(account_id_, region_, image_name, tag):    
    tagged_image_name=build_tagged_image(image_name, tag)
    password_stdin=build_ecr_password_stdin(account_id_, region_)
    
    return f"{password_stdin}/{tagged_image_name}"


def run(command):
    subprocess.run(command, stdout=DEVNULL, stderr=DEVNULL, shell=True)

def login_ecr_docker(account_id, region):
    print('Logining on ECR account...')

    password_stdin=build_ecr_password_stdin(account_id, region)
    
    opts=f"get-login-password --region {region}"
    get_pwd_command=f"aws ecr {opts}"
    
    opts=f"--username AWS --password-stdin {password_stdin}"
    login_command=f"docker login {opts}"
    
    entry_command=f"{get_pwd_command} | {login_command}"
    
    run(entry_command)
    
def create_ecr_image(ecr_image_name_):
    
    print('Creating ECR image...')
    
    args_1=f"--repository-name {ecr_image_name_}"
    args_2=f"--image-scanning-configuration scanOnPush=true"
    args_3=f"--image-tag-mutability MUTABLE"
    create_args=f"{args_1} {args_2} {args_3}"
    
    opts=f"create-repository {create_args}"
    create_comand=f"aws ecr {opts}"

    run(create_comand)

def delete_ecr_image(ecr_image_name_):
    print('Deleting existent ECR image...')

    tags=f"--force --repository-name {ecr_image_name_}"
    delete_command=f"aws ecr delete-repository {tags}" 
    
    run(delete_command)
    
def build_docker_image(ecr_image_name):
    print('Building docker image...')
    
    build_args=f"-q -t {ecr_image_name} ."
    build_command=f"docker build {build_args}"
    
    run(build_command)
    
def tag_docker_image(tagged_image_uri_, routed_url):
    print('Tagging docker image...')

    tag_args=f"{tagged_image_uri_} {routed_url}"
    tag_command=f"docker tag {tag_args}"
    
    run(tag_command)
    
def push_docker_image(tagged_image_uri):
    print('Pushing docker image to ECR...')

    push_command=f"docker push {tagged_image_uri}"

    run(push_command)
    
def pipe_push_image(account_id_, region_, ecr_image_name_, tag_):
    # 1. Log in to AWS ECR
    login_ecr_docker(account_id_, region_)
    
    # 2. Delete ECR repo: only needs to be done once
    delete_ecr_image(ecr_image_name_)
    
    # 3. Create ECR repo: only needs to be done once
    create_ecr_image(ecr_image_name_)

    # 4. Build Docker image using your local Dockerfile
    build_docker_image(ecr_image_name_)

    # 5. Tag you image
    tagged_image_uri=build_tagged_image(ecr_image_name_, tag_)
    routed_url=build_ecr_url(account_id_, region_, ecr_image_name_, tag_)
    tag_docker_image(tagged_image_uri, routed_url)

    # 6. Push your image to ECR
    push_docker_image(routed_url)