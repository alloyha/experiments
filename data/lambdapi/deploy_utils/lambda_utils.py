def get_function(l_client, function_name):
    failure_message=f"Lambda function {function_name} does not exist"
    
    try:
        return l_client.get_function(FunctionName=function_name)
    
    except l_client.exceptions.ResourceNotFoundException:
        print(failure_message)

def create_function(l_client, function_name, func_description, routed_url, role_arn):
    failure_message=f"Lambda function {function_name} already exists"
    
    code_payload={'ImageUri': routed_url}
    
    try:
        return l_client.create_function(
            FunctionName=function_name,
            Role=role_arn,
            PackageType='Image',
            Code=code_payload,
            Description=func_description,
            Timeout=10,
            MemorySize=256,
            Publish=True,
        )
    
    except l_client.exceptions.ResourceNotFoundException:
        print(failure_message)

# snippet-start:[python.example_code.lambda.DeleteFunction]
def delete_function(l_client, function_name):
    """
    Deletes a Lambda function.

    :param function_name: The name of the function to delete.
    """
    try:
        return l_client.delete_function(FunctionName=function_name)
    except l_client.exceptions.ClientError:
        print(f"Couldn't delete function {function_name}.", )