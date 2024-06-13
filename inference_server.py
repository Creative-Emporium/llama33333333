
from typing import List, Optional
import sys
import torch
import flask
import fire
import json

from llama import Dialog, Llama



app = flask.Flask(__name__)


@app.route('/chat-completions', methods=['POST'])
def chat ():
	request = flask.request
	print('request:', request.json)

	return flask.Response(json.dumps({
		'choices': [{
			'message': {
				'role': 'assistant',
				'content': 'emm',
			},
		}],
	}), mimetype='application/json')


def main (
		port: int = 8080,
		host: str = '127.0.0.1',
	):
	try:
		app.run(port=port, host=host, threaded=False)
	except:
		print('server interrupted:', sys.exc_info())



if __name__ == "__main__":
	fire.Fire(main)
