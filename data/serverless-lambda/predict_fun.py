import json
import cloudpickle

# By loading the pickle outside `predict`,
# we re-use it across different Lambda calls for the same execution instance
with open('model.pickle', 'rb') as f:
    model = cloudpickle.load(f)

def api_return(body, status):
    return {
        'isBase64Encoded': False,
        'statusCode': status,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(body, default=str)
    }

# Request input validation 
def validate_event(event, context): 
    body=event['body']

    # This is a custom elif statement
    if isinstance(body, float):
        payload = [body]

    elif isinstance(body, list):
        payload = body
    
    else:
        error_json={'error': 'Unknown input format'}
        code=400
        
        return api_return(error_json, code)
    
    # Scikit-learn needs a list or array as input
    if isinstance(payload, dict):
        payload = [payload]

    return payload

# Prediction
def predict(event, context):
    payload=validate_event(event, context)
        
    try:
        output = model.predict(payload).tolist()
    
    except Exception as e:
        error_json={'error': str(e)}
        code=500

        return api_return(error_json, code)

    return api_return(output, 200)