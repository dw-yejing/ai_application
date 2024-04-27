from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host='8.130.35.78', port="19530", user='root', password='Milvus#2023')
    print(f"\nList connections:")
    print(connections.list_connections())

def insert_data(name, data):
    create_connection()
    collection = Collection(name=name)
    collection.insert(data)
    collection.flush()


def search(name, search_vectors):
    create_connection()
    collection = Collection(name=name)
    collection.load()
    search_param = {
        "data": search_vectors,
        "anns_field": 'vec',
        "param": {"metric_type": 'L2', "params": {"nprobe": 16}},
        "limit": 5,
        "expr": "id >= 0"}
    results = collection.search(**search_param)
    idx = []
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            idx.append(res.id)
            print("Top {}: {}".format(j, res))
    return idx
