# Import the client from the 'elasticsearch' module
from elasticsearch import Elasticsearch

# Instantiate a client instance
client = Elasticsearch("http://localhost:9200")

# Call an API, in this example `info()`
resp = client.info()
