import pandas as pd
import phoenix as px
import argparse 

def add_arguments(parser):
    parser.add_argument('--input',type=str, help='input parquet file with labels, embeddings and other info')
    return parser 

def main(args):

	train_df = pd.read_parquet(args.input)
	print(train_df.head())

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
