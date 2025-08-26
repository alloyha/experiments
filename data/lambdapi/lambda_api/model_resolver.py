
# By loading the pickle outside `predict`,
# we re-use it across different Lambda calls
# for the same execution instance
#
# from cloudpickle import load
# with open('model.pickle', 'rb') as f:
#     model = load(f)

ALLOWED_TYPES = (int, float)

# REPLACE WITH:
#   - command call model.predict(payload).tolist()

def model_prediction_map(payload_list: list):
    def square_lambda(x):
        return x ** 2
    square_map = map(square_lambda, payload_list)

    return list(square_map)
