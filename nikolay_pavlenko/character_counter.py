from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from collections import Counter
import requests
import pathlib


def get_args() -> ArgumentParser:
	parser = ArgumentParser(description='Download file and count \
										unique characters.')

	parser.add_argument('url', help='Url where files are contained.', type=str)
	parser.add_argument('--num_processes', type=int, default=cpu_count(),
		help=f'CPU numbers for operation.{cpu_count()} CPU is used by default.')

	return parser.parse_args()


class CharacterCounter:
	def __init__(self, args: ArgumentParser, unique_file_name='files.txt',
			data_folder='files_folder', result_file='result.txt'):
		"""
		unique_file_name = The special file in which contains list
							of all files..(e.g 'files.txt')
		data_folder = The folder in which files will be downloaded.
		result_file = The file in which the number of characters will be written.
		"""

		self.SPECIAL_CHARACTERS = {
				'\t': 'TAB',
				' ': 'SPACE',
				'\n': 'NEW LINE'
		}

		self.args = args
		self.unique_file_name = unique_file_name
		self.result_file = result_file
		self.data_folder = pathlib.Path(data_folder)

		self.data_folder.mkdir(exist_ok=True)

	def _save_to_file(self, response, file_name):
		with open(f'{self.data_folder}/{file_name}', 'wb') as file:
			file.write(response.content)

	def _download_file_and_count(self, file_name: str) -> Counter:
		try:
			response = requests.get(self.args.url+file_name)
			response.raise_for_status()
			self._save_to_file(response, file_name)
			return Counter(response.content.decode('utf-8'))
		except requests.exceptions.HTTPError as e:
			print('HTTPError: ', e)
		except requests.exceptions.ConnectionError as e:
			print('ConnectionError: ', e)
		except requests.exceptions.RequestException as e:
			print('RequestException: ', e)
		except requests.exceptions.Timeout as e:
			print('TimeoutError: ', e)

	def _get_files_name(self) -> list:
		try:
			response = requests.get(self.args.url+self.unique_file_name)
			response.raise_for_status()
			files_name = response.content.decode('utf-8').strip().split()
			return files_name
		except requests.exceptions.HTTPError as e:
			print('HTTPError: ', e)
		except requests.exceptions.ConnectionError as e:
			print('ConnectionError: ', e)
		except requests.exceptions.RequestException as e:
			print('RequestException: ', e)
		except requests.exceptions.Timeout as e:
			print('TimeoutError: ', e)

	def main(self):
		with Pool(self.args.num_processes) as process:
			counter = process.map(self._download_file_and_count, self._get_files_name())

		with open(self.result_file, 'w', encoding='utf-8') as file:
			for key, value in sum(counter, Counter()).items():
				file.write(f'{self.SPECIAL_CHARACTERS.get(key, key)} {value}\n')


if __name__ == '__main__':
	CharacterCounter(get_args()).main()
