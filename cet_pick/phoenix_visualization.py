import pandas as pd
import phoenix as px
import argparse 
import socket

def check_port_in_use(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True
        
def find_next_available_port(start_port=7000, host='127.0.0.1'):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except socket.error:
                port += 1
                
def add_arguments(parser):
    parser.add_argument('--input',type=str, help='input parquet file with labels, embeddings and other info')
    parser.add_argument('--port', type=int, default=7000, help='port number to run the app')
    return parser 

def main(args):

	train_df = pd.read_parquet(args.input)
	print(train_df.head())
 
 	# Check if port is in use
	port = args.port
	DEFAULT_PORT = 7000
	
	if port != DEFAULT_PORT:
		if check_port_in_use(port):
			next_avail = find_next_available_port(start_port=port)
			print("Port {} is already in use. Next free port is: {}".format(port, next_avail))
			port = next_avail
   
		print("Using port {}".format(port))
		train_df['image'] = train_df['image'].str.replace(f'localhost:{DEFAULT_PORT}', f'localhost:{port}')
		# train_df.to_parquet(args.input, compression='gzip') # Optionally: Save the updated dataframe with new port

	train_schema = px.Schema(
		prediction_label_column_name="label",
		# feature_column_names = ['name','coord'],
		tag_column_names = ['name','coord'],
		embedding_feature_column_names = {
		"image_embedding": px.EmbeddingColumnNames(
			vector_column_name="embeddings",
			link_to_data_column_name="image")
		})

	train_ds = px.Dataset(dataframe=train_df, schema=train_schema)
	session = px.launch_app(train_ds)

if __name__=='__main__':
    parser = argparse.ArgumentParser('Script for interactive session')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
