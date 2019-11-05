import click

from . import *

@click.command()
@click.argument('torch_model')
@click.option('-o', '--output', required=True)
@click.option('-d', '--divider', default=1)
def main(torch_model, output, divider):
	import torch
	net = torch.load(torch_model)
	name = torch_model.rsplit('.',1)[0].rsplit('/',1)[1]
	s = sequential_to_str(net,name,divider)
	with open(output, 'w') as f:
		f.write(s)

main()
