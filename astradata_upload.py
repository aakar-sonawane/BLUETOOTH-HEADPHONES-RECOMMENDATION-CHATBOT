from MODULES.data_ingestion import data_ingestion
 
print("Starting fresh data ingestion... this may take a few minutes.\n")
 
vstore, insert_ids = data_ingestion(None)
 
print(f"\nInserted {len(insert_ids)} documents.\n")
 
print("--- Verifying metadata ---")
docs = vstore.similarity_search("BoAt Rockerz", k=3)
for d in docs:
    print(d.metadata)
 
print("\nDone! You can now run: python app.py")