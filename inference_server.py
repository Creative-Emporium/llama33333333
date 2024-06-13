
import sys
import os
import flask
import fire
import json

from llama import Dialog, Llama



app = flask.Flask(__name__)

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


@app.route('/chat-completions', methods=['POST'])
def chat ():
	request = flask.request
	#print('request:', request.json)

	dialogs = [request.json['messages'][-3:]]

	global g_generator, g_temperature, g_top_p
	results = g_generator.chat_completion(
		dialogs,
		max_gen_len=2400,
		temperature=g_temperature,
		top_p=g_top_p,
	)

	return flask.Response(json.dumps({
		'choices': [
			dict(message=result['generation'])
			for result in results
		],
	}), mimetype='application/json')


def main (
		ckpt_dir: str,
		tokenizer_path: str,
		port: int = 8080,
		host: str = '127.0.0.1',
		temperature: float = 0.6,
		top_p: float = 0.9,
	):
	global g_generator, g_temperature, g_top_p

	g_temperature = temperature
	g_top_p = top_p

	g_generator = Llama.build(
		ckpt_dir=ckpt_dir,
		tokenizer_path=tokenizer_path,
		max_seq_len=3072,
		max_batch_size=1,
	)

	try:
		app.run(port=port, host=host, threaded=False)
	except:
		print('server interrupted:', sys.exc_info())



if __name__ == "__main__":
	fire.Fire(main)
