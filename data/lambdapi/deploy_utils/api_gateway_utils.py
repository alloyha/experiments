def build_source_arn(region_, account_id_, rest_api_id_):
    return f'arn:aws:execute-api:{region_}:{account_id_}:{rest_api_id_}/*'

def build_lambda_uri(region_, lambda_arn_):
    uri_host=f"arn:aws:apigateway:{region_}:lambda:path"
    uri_route=f"2015-03-31/functions/{lambda_arn_}/invocations"
    return f"{uri_host}/{uri_route}"

def build_api_url(rest_api_id, region_, endpoint_, stage_):
    host=f"https://{rest_api_id}.execute-api.{region_}.amazonaws.com"
    route=f"{stage_}/{endpoint_}/"
    return f"{host}/{route}"

def has_api(g_client, rest_api_name_):
    response = g_client.get_rest_apis()
    create_api_on_gateway = True
    
    for item in response['items']:
        if item['name'] == rest_api_name_:
            create_api_on_gateway = False
            
    return create_api_on_gateway

def create_resource(g_client, rest_api_id_, endpoint_):
    response = g_client.get_resources(restApiId=rest_api_id_)
    root_id = response['items'][0]['id']
    
    response = g_client.create_resource(
        restApiId=rest_api_id_,
        parentId=root_id,
        pathPart=endpoint_,
    )
    
    resource_id = response['id']
    
    return resource_id

def create_rest_method(g_client, rest_api_id_, resource_id_, method_verb_):
    g_client.put_method(
        restApiId=rest_api_id_,
        resourceId=resource_id_,
        httpMethod=method_verb_,
        authorizationType='NONE', # WARNING: this will allow public access!
        apiKeyRequired=True,
    )

def create_rest_api(g_client, rest_api_name_):
    description='API Gateway that triggers a lambda function'
    response=g_client.create_rest_api(name=rest_api_name_, description=description) 
    
    rest_api_id = response['id']
    
    return rest_api_id

def get_lambda_arn(g_client, function_name_):
    response = g_client.get_function(FunctionName=function_name_)
    return response['Configuration']['FunctionArn']

def setup_integration(g_client, lambda_uri_, rest_api_id_, resource_id_, method_verb_):
    g_client.put_integration(
        restApiId=rest_api_id_,
        resourceId=resource_id_,
        httpMethod=method_verb_,
        type='AWS_PROXY',
        integrationHttpMethod=method_verb_,
        uri=lambda_uri_,
    )
    
def create_deployment(g_client, rest_api_id_, stage_):
    g_client.create_deployment(restApiId=rest_api_id_, stageName=stage_)
    
def create_api_key(g_client, rest_api_name_):
    response = g_client.create_api_key(
        name=rest_api_name_ + '-key',
        description='API key',
        enabled=True,
        generateDistinctId=True
    )
    
    api_key_id = response['id']
    api_key_value = response['value']
    
    return api_key_id, api_key_value

def create_usage_plan(g_client, rest_api_id_, stage_, usage_constraints_):
    name='API usage plan'
    description='Harsh rate limits and daily quota for public facing API'
    stages=[
        {
            'apiId': rest_api_id_,
            'stage': stage_,
        },
    ]
    
    response = g_client.create_usage_plan(
        name=name,
        description=description,
        apiStages=stages,
        throttle=usage_constraints_['rate_limits'],
        quota=usage_constraints_['quota']
    )
    
    usage_plan_id = response['id']
    
    return usage_plan_id

def create_usage_plan_key(g_client, usage_plan_id_, api_key_id_):
    g_client.create_usage_plan_key(
        usagePlanId=usage_plan_id_,
        keyId=api_key_id_,
        keyType='API_KEY'
    )
    
def add_apigateway_permission(l_client, function_name_, source_arn_):
    return l_client.add_permission(
        FunctionName=function_name_,
        StatementId='apigateway-lambda-invoke-permission',
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn=source_arn_
    )

def deploy_rest_api(\
        g_client, l_client, \
        account_id, region, \
        function_name_, rest_api_name_, endpoint_, method_verb_, \
        usage_constraints_, stage_ \
    ):
    # First, lets verify whether we already have an endpoint with this name.
    if not has_api(g_client, rest_api_name_):

        # 1. Create REST API
        rest_api_id = create_rest_api(g_client, rest_api_name_)

        # 2. Create resource
        resource_id=create_resource(g_client, rest_api_id, endpoint_)
        
        # 3. Create method
        create_rest_method(g_client, rest_api_id, resource_id, method_verb_)
        
        # 4. Get the Lambda function ARN
        lambda_arn = get_lambda_arn(l_client, function_name_)

        # 5. Set up integration with the Lambda function
        lambda_uri = build_lambda_uri(region, lambda_arn)

        setup_integration(g_client, lambda_uri, rest_api_id, resource_id, method_verb_)

        # 6. Deploy API
        create_deployment(g_client, rest_api_id, stage_)

        # 7. Create API key
        api_key_id, api_key_value = create_api_key(g_client, rest_api_name_)

        # 8. Create usage plan
        usage_plan_id = create_usage_plan(g_client, rest_api_id, stage_, usage_constraints_)
        
        # 9. Associate the usage plan with the API key
        create_usage_plan_key(g_client, usage_plan_id, api_key_id)
        
        # 10. Grant API Gateway permission to invoke the Lambda function
        source_arn = build_source_arn(region, account_id, rest_api_id)
        
        try: 
            add_apigateway_permission(l_client, function_name_, source_arn)
        except l_client.exceptions.ResourceConflictException:
            pass
        
        return {
            'url': build_api_url(rest_api_id, region, endpoint_, stage_),
            'api_key': api_key_value,
            'usage_plan_id': usage_plan_id,
            'rest_api_id': rest_api_id
        }
    
    else: 
        failure_msg=f"REST API name {rest_api_name_} is already under usage!"
        print(failure_msg)
        
        return {}