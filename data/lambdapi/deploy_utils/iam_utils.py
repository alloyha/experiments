from json import dumps

def try_get_role(i_client, role_name_, trust_policy):
    try:
        return i_client.get_role(
            RoleName=role_name_
        )
    except i_client.exceptions.NoSuchEntityException:
        i_client.create_role(
            RoleName=role_name_,
            AssumeRolePolicyDocument=dumps(trust_policy),
            Description='Execution role for Lambda function',
        )
        
def try_attach_role_policy(i_client, role_name_, policy_arn, trust_policy):
    # Just need to run it once, otherwise retrieve already existing role
    response=try_get_role(i_client, role_name_, trust_policy)

    # Get the role ARN
    role_arn = response['Role']['Arn']

    # Attach the AWSLambdaBasicExecutionRole policy to the role
    i_client.attach_role_policy(RoleName=role_name_, PolicyArn=policy_arn)
    
    return role_arn

