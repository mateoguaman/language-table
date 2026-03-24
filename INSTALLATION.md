# langtable environment
conda create -n langtable python=3.10
pip install --upgrade pip
pip install -r "language-table/requirements.txt"
pip install --no-deps git+https://github.com/google-research/scenic.git@ae21d9e884015aa7bc7cf1d489af53d16c249726