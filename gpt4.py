import getpass

from helm.common.authentication import Authentication
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.common.request import Request, RequestResult
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.proxy.accounts import Account
from helm.proxy.services.remote_service import RemoteService

# An example of how to use the request API.
auth = Authentication(api_key=open("gpt4_api_key.txt").read().strip())
# auth = Authentication(api_key="benchmarking-123")

service = RemoteService("https://crfm-models.stanford.edu")

# Access account and show my current quotas and usages
account: Account = service.get_account(auth)
print(account.usages["gpt4"])
import pdb; pdb.set_trace()

# # # Make a request
# request = Request(model="openai/gpt-4-0314", prompt="Life is like a box of")
# request_result: RequestResult = service.make_request(auth, request)
# print(request_result.completions[0].text)